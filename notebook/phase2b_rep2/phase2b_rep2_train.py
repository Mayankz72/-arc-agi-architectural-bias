"""
Phase 2b replication run 2: identical to notebook/phase2b/phase2b_train.py
except torch.manual_seed(1) -> torch.manual_seed(1) everywhere (different
random init + different data-shuffling order), to test whether Phase 2b's
result (RoPE-Transformer beating Mamba/hybrid ~6-7x, and showing the
project's first interpretable train_eval-vs-heldout gap) replicates under a
genuinely different seed rather than just a re-run of the same one. See
ROADMAP.md Phase 2b section for the result this is trying to replicate.

Phase 2b: corrected re-run of notebook/phase2/phase2_train.py.

The original Phase 2 pilot found the vanilla decoder-only Transformer produced
ZERO structurally valid grids at generation time, while Mamba/hybrid partially
worked. notebook/phase2_debug/ and notebook/phase2_debug2/ isolated the cause:
it was specifically the learned absolute positional embedding, not an
optimization issue and not exposure bias (see ROADMAP.md Phase 2 section).
Swapping in RoPE for self-attention alone took quickcheck_cell_acc from 0.000
to 0.204 in a 40-epoch debug run.

This script applies that fix to the real 3-architecture comparison:
  - transformer: self-attention now uses RoPE, no additive positional
    embedding at all (matches the debug2 config that worked).
  - mamba: unchanged from Phase 2 -- additive learned positional embedding +
    Mamba/SSM blocks, since this already worked fine.
  - hybrid: interleaved Mamba + Transformer blocks. The Transformer sub-blocks
    now use RoPE (same fix, since they're doing self-attention too); the
    additive positional embedding is kept for whatever benefit it gives the
    Mamba sub-blocks (unchanged from Phase 2). Caveat: this means hybrid's
    residual stream carries both an additive absolute-position signal AND
    RoPE-rotated attention -- an untested combination, not a perfectly clean
    ablation, but architecturally sensible (each mixing operator gets the
    positional scheme it actually needs) and worth flagging explicitly rather
    than hiding.
  - training regime upgraded for ALL THREE architectures (not just
    transformer) to the validated debug regime: lr=1e-3, warmup_frac=0.05,
    grad_clip=1.0, weight_decay=1e-2 -- so the comparison stays controlled
    (identical training regime across architectures), just using the better
    regime now that we know it doesn't hurt anything.
"""
import json
import subprocess
import sys
import time

# --- mamba-ssm install (see notebook/main.py env-check: needs --no-build-isolation) ---
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

PAD = VOCAB_SIZE  # 16, reserved purely for batching -- never a real content token
FULL_VOCAB = VOCAB_SIZE + 1
torch.manual_seed(1)

# --- data -------------------------------------------------------------------

DATA_DIR = f"{DATASET_ROOT}/data/synthetic"


def load_split(name):
    challenges = json.load(open(f"{DATA_DIR}/{name}_challenges.json"))
    solutions = json.load(open(f"{DATA_DIR}/{name}_solutions.json"))
    return challenges, solutions


def build_training_examples(challenges, solutions):
    """Leave-one-out augmentation: for a task with k train pairs + 1 known test
    pair, produce k+1 examples, each holding out one pair as the target and using
    all the others (train pairs plus the test pair, since we know its solution
    here) as context."""
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
        # only supervise positions predicting a target token, i.e. index j predicts
        # seq[j+1]; that's a target token iff (j+1) >= plen
        for j in range(len(lbl)):
            if j + 1 >= plen:
                labels[i, j] = lbl[j]
    return input_ids.to(DEVICE), labels.to(DEVICE)


# --- models -------------------------------------------------------------------

MAX_SEQ_LEN = 2048  # worst case: 3 context pairs, upscale2 doubling an 8x8 input to a 16x16 output (~1400 tokens)
ROPE_THETA = 10000.0
DROPOUT = 0.1


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttentionRoPE(nn.Module):
    """Manual QKV attention with rotary position embeddings -- the fix
    validated in notebook/phase2_debug2/. Used for every self-attention layer
    in this script (both the standalone transformer and hybrid's Transformer
    sub-blocks)."""

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
        # Pure self-attention now gets its positional info from RoPE (inside
        # CausalSelfAttentionRoPE) instead -- an additive absolute embedding on
        # top was exactly what crippled it (see ROADMAP.md). Mamba/hybrid keep
        # the additive embedding since it already worked for them.
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
        self.head.weight = self.embed.weight  # weight tying

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


# --- training / eval -----------------------------------------------------------

def train_model(kind, n_layers, train_examples, eval_challenges, eval_solutions, d_model=128, n_heads=4,
                 epochs=80, batch_size=32, lr=1e-3, warmup_frac=0.05, grad_clip=1.0, weight_decay=1e-2,
                 eval_every=20, eval_subsample=30):
    """eval_challenges/eval_solutions (a subsample of train_eval -- fresh instances of TRAINING
    combos, never the held-out ones) are used ONLY for checkpoint selection (early stopping
    against overfitting), never for gradient updates. The held-out compositional set is never
    touched until the very end, so model selection cannot leak into the actual research metric."""
    import copy

    torch.manual_seed(1)
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
            # >= (not >): on ties, prefer the LATER/more-trained checkpoint, never silently
            # fall back to an early near-untrained one just because nothing beat it outright.
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
    """Fraction of matching cells if shapes match; 0.0 if shapes differ (or pred invalid)."""
    if pred is None or len(pred) != len(gt) or any(len(pr) != len(gr) for pr, gr in zip(pred, gt)):
        return 0.0
    total = sum(len(row) for row in gt)
    correct = sum(1 for r_p, r_g in zip(pred, gt) for c_p, c_g in zip(r_p, r_g) if c_p == c_g)
    return correct / total if total else 0.0


@torch.no_grad()
def evaluate(model, challenges, solutions, kind_label, split_label, n_examples_to_print=2):
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
    print(f"[{kind_label}] {split_label}: exact_match={result['score']:.3f} "
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

    results = {}
    for kind in ["transformer", "mamba", "hybrid"]:
        model = train_model(kind, configs[kind], train_examples, train_eval_challenges, train_eval_solutions)
        results[kind] = {
            "n_params": count_params(model),
            "train_eval": evaluate(model, train_eval_challenges, train_eval_solutions, kind, "train_eval (in-distribution)"),
            "heldout": evaluate(model, heldout_challenges, heldout_solutions, kind, "heldout (compositional)"),
        }

    print("\n=== SUMMARY ===")
    for kind, r in results.items():
        te, ho = r["train_eval"], r["heldout"]
        print(f"{kind}: params={r['n_params']:,} "
              f"train_eval(exact={te['score']:.3f} cell={te['mean_cell_acc']:.3f} shape={te['shape_match_rate']:.3f}) "
              f"heldout(exact={ho['score']:.3f} cell={ho['mean_cell_acc']:.3f} shape={ho['shape_match_rate']:.3f})")

    with open("/kaggle/working/phase2b_rep2_results.json", "w") as f:
        json.dump(
            {k: {"n_params": v["n_params"],
                 "train_eval": {kk: vv for kk, vv in v["train_eval"].items() if kk != "per_task"},
                 "heldout": {kk: vv for kk, vv in v["heldout"].items() if kk != "per_task"},
                 "heldout_per_task": v["heldout"]["per_task"]} for k, v in results.items()},
            f,
        )
    print(f"\ntotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
