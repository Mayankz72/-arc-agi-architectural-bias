"""
Phase 3: NSA-style neuro-symbolic proposer+verifier layer, evaluated on top
of the same 3 matched-parameter architectures validated in Phase 2b.

Model/training code (Backbone, RoPE attention, train_model, generate, etc.)
is copied verbatim from notebook/phase2b/phase2b_train.py -- same seed(0),
same hyperparameters, same data -- so this run's own "baseline" column is a
fresh apples-to-apples reference for this run's verifier-augmented columns.
(It is a new training run, not a re-use of Phase 2b's checkpoints -- those
were never persisted outside the ephemeral Kaggle kernel -- so exact numbers
may differ slightly from Phase 2b's by run-to-run noise; the project already
has 2 independent seeds establishing the core Transformer>>Mamba/hybrid
finding, see ROADMAP.md Phase 2b.)

New in this script: after training, each architecture is evaluated under 4
conditions using arc_common.symbolic and arc_common.tta (see those modules'
docstrings for the full design rationale and honest caveats):
  1. baseline       -- greedy generation only (identical to Phase 2b's evaluate())
  2. tta_only       -- 8-view D4 test-time-augmentation majority vote, no symbolic
  3. symbolic_only  -- bounded (struct x color) program search, no neural model at
                       all (computed once, shared across all 3 architectures)
  4. full_pipeline  -- symbolic verifier first, falls back to TTA-ensembled
                       neural proposer only when no consistent program is found

Compute budgeting: tta_only requires 8x the generation calls of baseline, so
it runs on a fixed 50-task subsample per split (documented, not hidden) to
keep this within a single Kaggle GPU session; symbolic_only and full_pipeline
are run on the FULL splits since the symbolic search needs no GPU (full_pipeline
only falls through to the expensive neural+TTA path on tasks the symbolic
search can't solve, which -- per the local self-test in
test_phase3_harness.py -- is expected to be ~0% here).
"""
import json
import subprocess
import sys
import time

print("Installing mamba-ssm...")
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-q", "mamba-ssm", "causal-conv1d"],
    capture_output=True, text=True,
)
if r.returncode != 0:
    print(r.stdout[-3000:])
    print(r.stderr[-3000:])
    raise RuntimeError("mamba-ssm install failed")
print("mamba-ssm installed OK")

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

print("/kaggle/input contents:", os.listdir("/kaggle/input"))
DATASET_ROOT = None
for dirpath, dirnames, _ in os.walk("/kaggle/input"):
    if "arc_common" in dirnames:
        DATASET_ROOT = dirpath
        break
if DATASET_ROOT is None:
    raise RuntimeError("could not locate arc_common under /kaggle/input -- dataset not mounted as expected")
print("Using DATASET_ROOT:", DATASET_ROOT, "contents:", os.listdir(DATASET_ROOT))
sys.path.insert(0, DATASET_ROOT)
sys.path.insert(0, f"{DATASET_ROOT}/vendor/arc-dsl")

from arc_common.tokenizer import (  # noqa: E402
    BOS, EOS, GRID_OUT, VOCAB_SIZE, encode_prompt, encode_target, decode_prediction,
)
from arc_common.scoring import score_submission  # noqa: E402
from mamba_ssm import Mamba  # noqa: E402

# --- inlined arc_common.symbolic / arc_common.tta ----------------------------
# The shared Kaggle Dataset (kojijhjughio/arc-agi-inductive-bias-shared) has
# not been updated with these two new local-only modules (arc_common/symbolic.py,
# arc_common/tta.py) -- `kaggle datasets version` uploads go through a
# storage.googleapis.com resumable-upload endpoint that was silently
# unreachable (TLS handshake completes, zero bytes ever come back) from the
# network this was developed on. `kaggle kernels push` does not hit that same
# path and has worked reliably throughout this project, so rather than block
# Phase 3 on the dataset upload, this script inlines both modules' logic
# directly (verbatim, source-identical to arc_common/symbolic.py and
# arc_common/tta.py in the repo) so it only depends on the dataset version
# that's already live on Kaggle (arc_common/{tokenizer,scoring}.py,
# vendor/arc-dsl, data/synthetic, generator.py -- all unchanged).

from dsl import (  # noqa: E402
    identity, rot90, rot180, rot270, hmirror, vmirror, dmirror, cmirror,
    tophalf, bottomhalf, lefthalf, righthalf, trim, compress, switch, replace,
    upscale, downscale,
)
from collections import Counter  # noqa: E402

STRUCT_OPS = {
    "identity": identity,
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "hmirror": hmirror,
    "vmirror": vmirror,
    "dmirror": dmirror,
    "cmirror": cmirror,
    "tophalf": tophalf,
    "bottomhalf": bottomhalf,
    "lefthalf": lefthalf,
    "righthalf": righthalf,
    "trim": trim,
    "compress": compress,
    "upscale2": (lambda g: upscale(g, 2)),
    "upscale3": (lambda g: upscale(g, 3)),
    "downscale2": (lambda g: downscale(g, 2)),
}


def _to_tuple(grid):
    return tuple(tuple(row) for row in grid)


def _to_list(grid):
    if grid is None:
        return None
    grid = list(grid)
    if not grid or any(len(row) == 0 for row in grid):
        return None
    return [list(row) for row in grid]


def _color_ops(colors_present):
    yield "identity", (lambda g: g)
    colors = sorted(set(colors_present))
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            yield f"switch({a},{b})", (lambda g, a=a, b=b: switch(g, a, b))
    for a in colors:
        for b in colors:
            if a != b:
                yield f"replace({a},{b})", (lambda g, a=a, b=b: replace(g, a, b))


def find_consistent_programs(task, max_programs=1):
    demos = task["train"]
    if not demos:
        return []
    inputs_t = [_to_tuple(p["input"]) for p in demos]
    outputs_t = [_to_tuple(p["output"]) for p in demos]

    all_colors = set()
    for p in demos:
        for row in p["input"]:
            all_colors.update(row)
        for row in p["output"]:
            all_colors.update(row)
    all_colors_t = tuple(sorted(all_colors))

    found = []
    for struct_name, struct_fn in STRUCT_OPS.items():
        try:
            structured = [struct_fn(g) for g in inputs_t]
        except Exception:
            continue
        for color_name, color_fn in _color_ops(all_colors_t):
            try:
                ok = all(color_fn(s) == o for s, o in zip(structured, outputs_t))
            except Exception:
                continue
            if ok:
                found.append({"struct": struct_name, "color": color_name, "_color_fn": color_fn})
                if len(found) >= max_programs:
                    return found
    return found


def apply_program(program, input_grid):
    try:
        g = STRUCT_OPS[program["struct"]](_to_tuple(input_grid))
        g = program["_color_fn"](g)
    except Exception:
        return None
    return _to_list(g)


def symbolic_predict(task, test_index=0):
    programs = find_consistent_programs(task, max_programs=1)
    if not programs:
        return None
    return apply_program(programs[0], task["test"][test_index]["input"])


def _wrap_d4(fn):
    def wrapped(grid):
        g = tuple(tuple(row) for row in grid)
        return [list(row) for row in fn(g)]
    return wrapped


D4 = [
    ("identity", _wrap_d4(identity), _wrap_d4(identity)),
    ("rot90", _wrap_d4(rot90), _wrap_d4(rot270)),
    ("rot180", _wrap_d4(rot180), _wrap_d4(rot180)),
    ("rot270", _wrap_d4(rot270), _wrap_d4(rot90)),
    ("hmirror", _wrap_d4(hmirror), _wrap_d4(hmirror)),
    ("vmirror", _wrap_d4(vmirror), _wrap_d4(vmirror)),
    ("dmirror", _wrap_d4(dmirror), _wrap_d4(dmirror)),
    ("cmirror", _wrap_d4(cmirror), _wrap_d4(cmirror)),
]


def transform_task(task, fwd):
    return {
        "train": [{"input": fwd(p["input"]), "output": fwd(p["output"])} for p in task["train"]],
        "test": [{"input": fwd(t["input"])} for t in task["test"]],
    }


def _grid_key(grid):
    return tuple(tuple(row) for row in grid)


def tta_predict(task, test_index, predict_fn, transforms=D4):
    candidates = []
    for name, fwd, inv in transforms:
        try:
            t_task = transform_task(task, fwd)
            pred = predict_fn(t_task, test_index)
            if pred is None:
                continue
            candidates.append((name, inv(pred)))
        except Exception:
            continue
    if not candidates:
        return None, []
    counts = Counter(_grid_key(g) for _, g in candidates)
    best_count = counts.most_common(1)[0][1]
    tied_keys = {k for k, c in counts.items() if c == best_count}
    for name, g in candidates:
        if name == "identity" and _grid_key(g) in tied_keys:
            return g, candidates
    for _, g in candidates:
        if _grid_key(g) in tied_keys:
            return g, candidates
    return candidates[0][1], candidates


def combined_predict(task, test_index, neural_predict_fn, use_tta=True, use_symbolic=True):
    if use_symbolic:
        sym = symbolic_predict(task, test_index)
        if sym is not None:
            return sym, "symbolic"
    if use_tta:
        pred, _ = tta_predict(task, test_index, neural_predict_fn)
    else:
        pred = neural_predict_fn(task, test_index)
    return pred, "neural"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

PAD = VOCAB_SIZE
FULL_VOCAB = VOCAB_SIZE + 1
torch.manual_seed(0)

# --- data -------------------------------------------------------------------

DATA_DIR = f"{DATASET_ROOT}/data/synthetic"


def load_split(name):
    challenges = json.load(open(f"{DATA_DIR}/{name}_challenges.json"))
    solutions = json.load(open(f"{DATA_DIR}/{name}_solutions.json"))
    return challenges, solutions


def build_training_examples(challenges, solutions):
    examples = []
    for task_id, task in challenges.items():
        all_pairs = list(task["train"]) + [
            {"input": task["test"][i]["input"], "output": solutions[task_id][i]}
            for i in range(len(task["test"]))
        ]
        for held_idx in range(len(all_pairs)):
            context = all_pairs[:held_idx] + all_pairs[held_idx + 1:]
            target_pair = all_pairs[held_idx]
            pseudo_task = {"train": context, "test": [{"input": target_pair["input"]}]}
            prompt = encode_prompt(pseudo_task, test_index=0)
            target = encode_target(target_pair["output"])
            examples.append((prompt, target))
    return examples


def collate(batch):
    seqs = [p + t for p, t in batch]
    prompt_lens = [len(p) for p, _ in batch]
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len - 1), PAD, dtype=torch.long)
    labels = torch.full((len(seqs), max_len - 1), -100, dtype=torch.long)
    for i, (seq, plen) in enumerate(zip(seqs, prompt_lens)):
        inp = seq[:-1]
        lbl = seq[1:]
        input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        for j in range(len(lbl)):
            if j + 1 >= plen:
                labels[i, j] = lbl[j]
    return input_ids.to(DEVICE), labels.to(DEVICE)


# --- models (verbatim from phase2b_train.py) --------------------------------

MAX_SEQ_LEN = 2048
ROPE_THETA = 10000.0
DROPOUT = 0.1


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=MAX_SEQ_LEN, theta=ROPE_THETA):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos = self.cos_cached[:L].to(x.dtype)[None, None, :, :]
        sin = self.sin_cached[:L].to(x.dtype)[None, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionRoPE(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return x + self.drop(self.mamba(self.ln(x)))


class Backbone(nn.Module):
    def __init__(self, kind, d_model, n_layers, n_heads=4, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.embed = nn.Embedding(FULL_VOCAB, d_model)
        self.use_pos_embed = kind != "transformer"
        if self.use_pos_embed:
            self.pos_embed = nn.Embedding(MAX_SEQ_LEN, d_model)
        if kind == "transformer":
            layers = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        elif kind == "mamba":
            layers = [MambaBlock(d_model) for _ in range(n_layers)]
        elif kind == "hybrid":
            layers = []
            for i in range(n_layers):
                layers.append(MambaBlock(d_model) if i % 2 == 0 else TransformerBlock(d_model, n_heads, d_ff))
        else:
            raise ValueError(kind)
        self.layers = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, FULL_VOCAB, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        if self.use_pos_embed:
            L = input_ids.size(1)
            positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def calibrate(d_model=128, n_heads=4, target_kind="transformer", target_layers=4):
    ref = Backbone(target_kind, d_model, target_layers, n_heads)
    target = count_params(ref)
    print(f"reference ({target_kind}, {target_layers} layers): {target:,} params")
    configs = {"transformer": target_layers}
    for kind in ["mamba", "hybrid"]:
        best_layers, best_diff = None, float("inf")
        for n_layers in range(2, 25):
            m = Backbone(kind, d_model, n_layers, n_heads)
            diff = abs(count_params(m) - target)
            if diff < best_diff:
                best_diff, best_layers = diff, n_layers
            del m
        configs[kind] = best_layers
        m = Backbone(kind, d_model, best_layers, n_heads)
        print(f"{kind}: {best_layers} layers -> {count_params(m):,} params (target {target:,})")
    return configs


# --- training (verbatim from phase2b_train.py) -------------------------------

def train_model(kind, n_layers, train_examples, eval_challenges, eval_solutions, d_model=128, n_heads=4,
                 epochs=80, batch_size=32, lr=1e-3, warmup_frac=0.05, grad_clip=1.0, weight_decay=1e-2,
                 eval_every=20, eval_subsample=30):
    import copy

    torch.manual_seed(0)
    model = Backbone(kind, d_model, n_layers, n_heads).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(train_examples)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"\n=== training {kind} ({count_params(model):,} params) on {n} examples ===")

    sample_ids = list(eval_challenges)[:eval_subsample]
    sub_challenges = {k: eval_challenges[k] for k in sample_ids}
    sub_solutions = {k: eval_solutions[k] for k in sample_ids}

    best_score, best_state, best_epoch = -1.0, None, -1
    for epoch in range(epochs):
        perm = torch.randperm(n).tolist()
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            batch = [train_examples[j] for j in perm[i:i + batch_size]]
            input_ids, labels = collate(batch)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, FULL_VOCAB), labels.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            total_loss += loss.item()
            n_batches += 1
        do_eval = epoch % eval_every == 0 or epoch == epochs - 1
        if do_eval:
            score = quick_cell_acc(model, sub_challenges, sub_solutions)
            cur_lr = opt.param_groups[0]["lr"]
            print(f"  epoch {epoch}: loss {total_loss / n_batches:.4f}  lr {cur_lr:.2e}  quickcheck_cell_acc {score:.3f}")
            if score >= best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
    print(f"  best checkpoint: epoch {best_epoch} (quickcheck_cell_acc {best_score:.3f})")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def quick_cell_acc(model, challenges, solutions):
    model.eval()
    accs = []
    for task_id, task in challenges.items():
        for i in range(len(task["test"])):
            prompt = encode_prompt(task, test_index=i)
            pred = decode_prediction(generate(model, prompt))
            accs.append(cell_accuracy(pred, solutions[task_id][i]))
    model.train()
    return sum(accs) / len(accs) if accs else 0.0


@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens=300):
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)
    for _ in range(max_new_tokens):
        logits = model(ids)
        next_id = int(logits[0, -1].argmax())
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        if next_id == EOS:
            break
    return ids[0, len(prompt_tokens):].tolist()


def cell_accuracy(pred, gt):
    if pred is None or len(pred) != len(gt) or any(len(pr) != len(gr) for pr, gr in zip(pred, gt)):
        return 0.0
    total = sum(len(row) for row in gt)
    correct = sum(1 for r_p, r_g in zip(pred, gt) for c_p, c_g in zip(r_p, r_g) if c_p == c_g)
    return correct / total if total else 0.0


@torch.no_grad()
def evaluate(model, challenges, solutions, kind_label, split_label, n_examples_to_print=2):
    """Baseline condition: identical to phase2b's evaluate() -- greedy generation, no verifier."""
    model.eval()
    submission = {}
    cell_accs, shape_matches, printed = [], 0, 0
    for task_id, task in challenges.items():
        entries = []
        for i in range(len(task["test"])):
            prompt = encode_prompt(task, test_index=i)
            gen = generate(model, prompt)
            pred = decode_prediction(gen)
            gt = solutions[task_id][i]
            cell_accs.append(cell_accuracy(pred, gt))
            if pred is not None and len(pred) == len(gt) and all(len(pr) == len(gr) for pr, gr in zip(pred, gt)):
                shape_matches += 1
            if printed < n_examples_to_print:
                print(f"  [{kind_label}/{split_label}] example {task_id}: gt_shape={len(gt)}x{len(gt[0])} "
                      f"pred_shape={'invalid' if pred is None else f'{len(pred)}x{len(pred[0])}'} "
                      f"cell_acc={cell_accs[-1]:.2f}")
                printed += 1
            attempt = pred if pred is not None else [[0]]
            entries.append({"attempt_1": attempt, "attempt_2": attempt})
        submission[task_id] = entries
    result = score_submission(submission, solutions)
    model.train()
    n = len(cell_accs)
    result["mean_cell_acc"] = sum(cell_accs) / n
    result["shape_match_rate"] = shape_matches / n
    print(f"[{kind_label}] {split_label} (baseline): exact_match={result['score']:.3f} "
          f"mean_cell_acc={result['mean_cell_acc']:.3f} shape_match_rate={result['shape_match_rate']:.3f}")
    return result


# --- Phase 3: verifier-augmented evaluation ----------------------------------

@torch.no_grad()
def evaluate_with_predict_fn(model, challenges, solutions, kind_label, split_label, condition_label, predict_fn):
    """Generic evaluator for any predict_fn(task, test_index) -> grid|None,
    e.g. a closure around tta_predict/combined_predict. Same scoring path
    (score_submission) and metrics (mean_cell_acc, shape_match_rate) as the
    baseline evaluate() above, so all 4 conditions are directly comparable."""
    model.eval()
    submission = {}
    cell_accs, shape_matches = [], 0
    for task_id, task in challenges.items():
        entries = []
        for i in range(len(task["test"])):
            pred = predict_fn(task, i)
            gt = solutions[task_id][i]
            cell_accs.append(cell_accuracy(pred, gt))
            if pred is not None and len(pred) == len(gt) and all(len(pr) == len(gr) for pr, gr in zip(pred, gt)):
                shape_matches += 1
            attempt = pred if pred is not None else [[0]]
            entries.append({"attempt_1": attempt, "attempt_2": attempt})
        submission[task_id] = entries
    result = score_submission(submission, solutions)
    model.train()
    n = len(cell_accs)
    result["mean_cell_acc"] = sum(cell_accs) / n
    result["shape_match_rate"] = shape_matches / n
    print(f"[{kind_label}] {split_label} ({condition_label}): exact_match={result['score']:.3f} "
          f"mean_cell_acc={result['mean_cell_acc']:.3f} shape_match_rate={result['shape_match_rate']:.3f}")
    return result


def make_neural_predict_fn(model):
    def predict(task, test_index):
        prompt = encode_prompt(task, test_index=test_index)
        return decode_prediction(generate(model, prompt))
    return predict


def subsample(challenges, solutions, n):
    ids = list(challenges)[:n]
    return {k: challenges[k] for k in ids}, {k: solutions[k] for k in ids}


TTA_SUBSAMPLE = 50  # see module docstring: bounds the 8x-generation-cost tta_only condition


@torch.no_grad()
def evaluate_symbolic_only(challenges, solutions, split_label):
    """Independent of any trained model -- computed once, shared across architectures."""
    submission = {}
    cell_accs, shape_matches = [], 0
    for task_id, task in challenges.items():
        entries = []
        for i in range(len(task["test"])):
            pred = symbolic_predict(task, i)
            gt = solutions[task_id][i]
            cell_accs.append(cell_accuracy(pred, gt))
            if pred is not None and len(pred) == len(gt) and all(len(pr) == len(gr) for pr, gr in zip(pred, gt)):
                shape_matches += 1
            attempt = pred if pred is not None else [[0]]
            entries.append({"attempt_1": attempt, "attempt_2": attempt})
        submission[task_id] = entries
    result = score_submission(submission, solutions)
    n = len(cell_accs)
    result["mean_cell_acc"] = sum(cell_accs) / n
    result["shape_match_rate"] = shape_matches / n
    print(f"[symbolic_only] {split_label}: exact_match={result['score']:.3f} "
          f"mean_cell_acc={result['mean_cell_acc']:.3f} shape_match_rate={result['shape_match_rate']:.3f}")
    return result


def main():
    t0 = time.time()
    train_challenges, train_solutions = load_split("train")
    train_eval_challenges, train_eval_solutions = load_split("train_eval")
    heldout_challenges, heldout_solutions = load_split("heldout")

    train_examples = build_training_examples(train_challenges, train_solutions)
    print(f"built {len(train_examples)} leave-one-out training examples")

    configs = calibrate()

    # symbolic_only doesn't depend on the neural model -- compute once
    print("\n=== symbolic_only (shared across all architectures) ===")
    symbolic_results = {
        "train_eval": evaluate_symbolic_only(train_eval_challenges, train_eval_solutions, "train_eval (in-distribution)"),
        "heldout": evaluate_symbolic_only(heldout_challenges, heldout_solutions, "heldout (compositional)"),
    }

    te_sub_c, te_sub_s = subsample(train_eval_challenges, train_eval_solutions, TTA_SUBSAMPLE)
    ho_sub_c, ho_sub_s = subsample(heldout_challenges, heldout_solutions, TTA_SUBSAMPLE)

    results = {}
    for kind in ["transformer", "mamba", "hybrid"]:
        model = train_model(kind, configs[kind], train_examples, train_eval_challenges, train_eval_solutions)
        neural_predict = make_neural_predict_fn(model)

        baseline_te = evaluate(model, train_eval_challenges, train_eval_solutions, kind, "train_eval (in-distribution)")
        baseline_ho = evaluate(model, heldout_challenges, heldout_solutions, kind, "heldout (compositional)")

        tta_te = evaluate_with_predict_fn(
            model, te_sub_c, te_sub_s, kind, f"train_eval (subsample={TTA_SUBSAMPLE})", "tta_only",
            lambda t, i: tta_predict(t, i, neural_predict)[0],
        )
        tta_ho = evaluate_with_predict_fn(
            model, ho_sub_c, ho_sub_s, kind, f"heldout (subsample={TTA_SUBSAMPLE})", "tta_only",
            lambda t, i: tta_predict(t, i, neural_predict)[0],
        )

        full_te = evaluate_with_predict_fn(
            model, train_eval_challenges, train_eval_solutions, kind, "train_eval (in-distribution)", "full_pipeline",
            lambda t, i: combined_predict(t, i, neural_predict, use_tta=True, use_symbolic=True)[0],
        )
        full_ho = evaluate_with_predict_fn(
            model, heldout_challenges, heldout_solutions, kind, "heldout (compositional)", "full_pipeline",
            lambda t, i: combined_predict(t, i, neural_predict, use_tta=True, use_symbolic=True)[0],
        )

        results[kind] = {
            "n_params": count_params(model),
            "baseline": {"train_eval": baseline_te, "heldout": baseline_ho},
            "tta_only": {"train_eval": tta_te, "heldout": tta_ho},
            "full_pipeline": {"train_eval": full_te, "heldout": full_ho},
        }

    print("\n=== SUMMARY ===")
    print(f"symbolic_only: train_eval(exact={symbolic_results['train_eval']['score']:.3f}) "
          f"heldout(exact={symbolic_results['heldout']['score']:.3f})")
    for kind, r in results.items():
        b_te, b_ho = r["baseline"]["train_eval"], r["baseline"]["heldout"]
        t_te, t_ho = r["tta_only"]["train_eval"], r["tta_only"]["heldout"]
        f_te, f_ho = r["full_pipeline"]["train_eval"], r["full_pipeline"]["heldout"]
        print(f"{kind}: params={r['n_params']:,}")
        print(f"  baseline      train_eval(cell={b_te['mean_cell_acc']:.3f}) heldout(cell={b_ho['mean_cell_acc']:.3f})")
        print(f"  tta_only      train_eval(cell={t_te['mean_cell_acc']:.3f}) heldout(cell={t_ho['mean_cell_acc']:.3f})")
        print(f"  full_pipeline train_eval(exact={f_te['score']:.3f} cell={f_te['mean_cell_acc']:.3f}) "
              f"heldout(exact={f_ho['score']:.3f} cell={f_ho['mean_cell_acc']:.3f})")

    def strip_per_task(d):
        return {kk: vv for kk, vv in d.items() if kk != "per_task"}

    out = {
        "symbolic_only": {k: strip_per_task(v) for k, v in symbolic_results.items()},
        "architectures": {
            kind: {
                "n_params": r["n_params"],
                "baseline": {k: strip_per_task(v) for k, v in r["baseline"].items()},
                "tta_only": {k: strip_per_task(v) for k, v in r["tta_only"].items()},
                "full_pipeline": {k: strip_per_task(v) for k, v in r["full_pipeline"].items()},
            }
            for kind, r in results.items()
        },
    }
    with open("/kaggle/working/phase3_results.json", "w") as f:
        json.dump(out, f)
    print(f"\ntotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
