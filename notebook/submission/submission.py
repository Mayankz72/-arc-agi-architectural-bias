"""
Phase 5: the actual Kaggle competition submission notebook. Produces
/kaggle/working/submission.json for the 240-task blind test set
(arc-agi_test_challenges.json), in the exact format the competition requires.

Design decisions, and why:
  - Transformer (RoPE) ONLY, no Mamba/hybrid. Two independent reasons converge
    on this: (1) it's the architecture every experiment in this project (Phase
    2b, 2c, 3, 4a, 4b) found to be the strongest and the most interpretable --
    submitting the model the paper's own findings say is best keeps the paper
    and the submission methodologically consistent, rather than submitting a
    different model than the one being written about; (2) `mamba-ssm` needs a
    `pip install` at kernel start, which needs internet access -- but a real
    scored competition run has NO internet access. Dropping Mamba/hybrid
    entirely sidesteps that constraint rather than working around it.
  - Same ~800K-parameter scale used throughout the whole project (d_model=128,
    4 layers), not scaled up. The "properly resourced" lever here is more
    TRAINING TIME on real data (a wall-clock training budget, see below), not
    a bigger model -- consistent with how Phase 4 already framed "properly
    resourced" (more epochs/data, not bigger models), and keeps this
    submission's architecture identical to what the paper reports on.
  - FILTER_CAP=8192 (vs. Phase 4a's 4096): measured locally before writing
    this script -- at 8192, training/public-eval/blind-test coverage is
    99.9%/98.3%/100.0% (vs. 94.3%/80.0%/-- at 4096). The blind test set is
    what's actually being predicted here, so covering all 240 of its tasks
    matters more than it did for Phase 4a's research-only eval.
  - Wall-clock-bounded training (MAX_TRAIN_SECONDS), not a fixed epoch count:
    this cap is unprecedented at this project's scale (2x Phase 4a's), so
    per-epoch cost isn't precisely known in advance. A wall-clock stop
    guarantees this finishes within Kaggle's runtime limit with time left for
    inference, whatever the actual per-epoch cost turns out to be.
  - Uses the KV-cached generate() (arc_common/models.py's forward_prefill/
    forward_step, inlined the same way Phase 4a did) since predictions must be
    generated for the blind test set's largest prompts (up to 6,515 raw
    tokens) plus up to MAX_NEW_TOKENS more -- the old un-cached generate()
    would be prohibitively slow here (this is exactly the mechanism Phase 4a
    was built to fix).
  - Symbolic verifier (from Phase 3, unchanged) runs first on every test
    example; falls back to the trained neural model only when no consistent
    program is found. Free accuracy on any test task that happens to fall
    inside the narrow curated struct+color DSL, with zero downside otherwise.
"""
import json
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

print("/kaggle/input contents:", os.listdir("/kaggle/input"))
DATASET_ROOT = None
COMPETITION_ROOT = None
for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
    if "arc_common" in dirnames and DATASET_ROOT is None:
        DATASET_ROOT = dirpath
    if "arc-agi_training_challenges.json" in filenames and COMPETITION_ROOT is None:
        COMPETITION_ROOT = dirpath
if DATASET_ROOT is None:
    raise RuntimeError("could not locate arc_common under /kaggle/input -- shared-code dataset not mounted")
if COMPETITION_ROOT is None:
    raise RuntimeError("could not locate arc-agi_training_challenges.json under /kaggle/input -- "
                        "competition data not mounted (check kernel-metadata.json's competition_sources)")
print("Using DATASET_ROOT:", DATASET_ROOT)
print("Using COMPETITION_ROOT:", COMPETITION_ROOT, "contents:", os.listdir(COMPETITION_ROOT))

import sys  # noqa: E402
sys.path.insert(0, DATASET_ROOT)
sys.path.insert(0, f"{DATASET_ROOT}/vendor/arc-dsl")

from arc_common.tokenizer import (  # noqa: E402
    BOS, EOS, GRID_OUT, VOCAB_SIZE, encode_prompt, encode_target, decode_prediction,
)
from arc_common.scoring import build_submission_template, validate_submission, score_submission  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

PAD = VOCAB_SIZE
FULL_VOCAB = VOCAB_SIZE + 1
torch.manual_seed(0)

# --- inlined arc_common.symbolic (verifier), unchanged from Phase 3/4a --------

from dsl import (  # noqa: E402
    identity, rot90, rot180, rot270, hmirror, vmirror, dmirror, cmirror,
    tophalf, bottomhalf, lefthalf, righthalf, trim, compress, switch, replace,
    upscale, downscale,
)

STRUCT_OPS = {
    "identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270,
    "hmirror": hmirror, "vmirror": vmirror, "dmirror": dmirror, "cmirror": cmirror,
    "tophalf": tophalf, "bottomhalf": bottomhalf, "lefthalf": lefthalf, "righthalf": righthalf,
    "trim": trim, "compress": compress,
    "upscale2": (lambda g: upscale(g, 2)), "upscale3": (lambda g: upscale(g, 3)),
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


# --- inlined arc_common.models (Transformer-only; no Mamba/hybrid, see docstring) ---

FILTER_CAP = 8192
MAX_NEW_TOKENS = 1024
MAX_SEQ_LEN = FILTER_CAP + MAX_NEW_TOKENS + 2048  # headroom for the RoPE cache -- a 128-token
# margin (used in the first real run, 2026-09-08) turned out too tight: one public-eval task's
# prompt alone exceeded FILTER_CAP (only TRAINING examples are length-filtered in this script --
# eval/blind-test prompts are never dropped, since every test task needs a prediction), pushing
# the total prefill length to 9,345 against a 9,344-slot cache and raising a shape-mismatch
# error. It was caught by the existing try/except fallback (no crash, no invalid submission),
# but a much larger margin removes the failure mode instead of just tolerating it.
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
        assert self.head_dim % 2 == 0
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _qkv(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _rope(self, q, k, start_pos, length):
        cos = self.cos_cached[start_pos:start_pos + length].to(q.dtype)[None, None, :, :]
        sin = self.sin_cached[start_pos:start_pos + length].to(q.dtype)[None, None, :, :]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self._qkv(x)
        q, k = self._rope(q, k, 0, L)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

    def forward_prefill(self, x):
        B, L, D = x.shape
        q, k, v = self._qkv(x)
        q, k = self._rope(q, k, 0, L)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), (k, v)

    def forward_step(self, x, cache, position):
        q, k, v = self._qkv(x)
        q, k = self._rope(q, k, position, 1)
        if cache is None:
            k_cache, v_cache = k, v
        else:
            k_cache = torch.cat([cache[0], k], dim=2)
            v_cache = torch.cat([cache[1], v], dim=2)
        out = F.scaled_dot_product_attention(q, k_cache, v_cache, is_causal=False)
        out = out.transpose(1, 2).reshape(x.shape[0], 1, -1)
        return self.out_proj(out), (k_cache, v_cache)


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

    def forward_prefill(self, x):
        attn_out, cache = self.attn.forward_prefill(self.ln1(x))
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, cache

    def forward_step(self, x, cache, position):
        attn_out, cache = self.attn.forward_step(self.ln1(x), cache, position)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, cache


class Backbone(nn.Module):
    """Transformer-only (no use_pos_embed path -- RoPE handles position, see
    Phase 2 debug findings: an additive absolute positional embedding on top
    of self-attention is what crippled the Transformer in the first place)."""

    def __init__(self, d_model, n_layers, n_heads=4, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.embed = nn.Embedding(FULL_VOCAB, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, FULL_VOCAB, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)

    def forward_prefill(self, input_ids):
        x = self.embed(input_ids)
        cache = []
        for layer in self.layers:
            x, layer_cache = layer.forward_prefill(x)
            cache.append(layer_cache)
        x = self.ln_f(x)
        return self.head(x), cache

    def forward_step(self, new_token_ids, cache, position):
        x = self.embed(new_token_ids)
        new_cache = []
        for layer, layer_cache in zip(self.layers, cache):
            x, updated = layer.forward_step(x, layer_cache, position)
            new_cache.append(updated)
        x = self.ln_f(x)
        return self.head(x), new_cache


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens=MAX_NEW_TOKENS):
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)
    logits, cache = model.forward_prefill(ids)
    next_id = int(logits[0, -1].argmax())
    generated = [next_id]
    position = ids.size(1)
    if next_id != EOS:
        for _ in range(max_new_tokens - 1):
            new_tok = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
            logits, cache = model.forward_step(new_tok, cache, position)
            next_id = int(logits[0, -1].argmax())
            generated.append(next_id)
            position += 1
            if next_id == EOS:
                break
    return generated


# --- data ---------------------------------------------------------------------

def load_real(prefix, with_solutions=True):
    challenges = json.load(open(f"{COMPETITION_ROOT}/{prefix}_challenges.json"))
    solutions = json.load(open(f"{COMPETITION_ROOT}/{prefix}_solutions.json")) if with_solutions else None
    return challenges, solutions


def build_training_examples(challenges, solutions, cap):
    """Leave-one-out augmentation; filters individual derived examples by
    length (not whole tasks -- see Phase 4a's PROGRESS.md entry for why:
    holding out a train pair pulls the test pair into that example's
    context, so it can exceed a cap the task's own eval prompt satisfies)."""
    examples = []
    dropped = 0
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
            if len(prompt) + len(target) <= cap:
                examples.append((prompt, target))
            else:
                dropped += 1
    print(f"build_training_examples: kept {len(examples)}, dropped {dropped} oversized (> {cap} tokens)")
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


def cell_accuracy(pred, gt):
    if pred is None or len(pred) != len(gt) or any(len(pr) != len(gr) for pr, gr in zip(pred, gt)):
        return 0.0
    total = sum(len(row) for row in gt)
    correct = sum(1 for r_p, r_g in zip(pred, gt) for c_p, c_g in zip(r_p, r_g) if c_p == c_g)
    return correct / total if total else 0.0


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


# --- training: wall-clock-bounded, not a fixed epoch count --------------------

MAX_TRAIN_SECONDS = 6 * 3600  # 6h training budget, leaving generous headroom under Kaggle's
                              # 12h limit for setup + inference + notebook conversion overhead


def train_model(train_examples, eval_challenges, eval_solutions, d_model=128, n_layers=4, n_heads=4,
                 batch_size=8, lr=1e-3, warmup_frac=0.02, grad_clip=1.0, weight_decay=1e-2,
                 eval_every_seconds=900, eval_subsample=20, max_train_seconds=MAX_TRAIN_SECONDS):
    torch.manual_seed(0)
    model = Backbone(d_model, n_layers, n_heads).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(train_examples)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    # warmup based on an estimate of total steps at a nominal 150-epoch budget -- if the
    # wall-clock stops training earlier or later than that estimate, warmup still completes
    # early in the run either way (warmup_frac is small), so this doesn't need to be exact.
    warmup_steps = max(1, int(steps_per_epoch * 150 * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"training Transformer ({count_params(model):,} params) on {n} examples, "
          f"{steps_per_epoch} steps/epoch, max_train_seconds={max_train_seconds}")

    sample_ids = list(eval_challenges)[:eval_subsample]
    sub_challenges = {k: eval_challenges[k] for k in sample_ids}
    sub_solutions = {k: eval_solutions[k] for k in sample_ids}

    best_score, best_state, best_epoch = -1.0, None, -1
    t_start = time.time()
    last_eval_time = t_start
    epoch = 0
    while True:
        elapsed = time.time() - t_start
        if elapsed >= max_train_seconds:
            print(f"  stopping: max_train_seconds ({max_train_seconds}s) reached at epoch {epoch}")
            break
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
        now = time.time()
        if now - last_eval_time >= eval_every_seconds or now - t_start >= max_train_seconds:
            score = quick_cell_acc(model, sub_challenges, sub_solutions)
            cur_lr = opt.param_groups[0]["lr"]
            print(f"  epoch {epoch}: t={now - t_start:.0f}s loss {total_loss / n_batches:.4f} "
                  f"lr {cur_lr:.2e} quickcheck_cell_acc {score:.3f}")
            last_eval_time = now
            if score >= best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
        epoch += 1
    print(f"  best checkpoint: epoch {best_epoch} (quickcheck_cell_acc {best_score:.3f}), "
          f"total epochs run: {epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def make_verifier_predict_fn(model):
    def predict(task, i):
        sym = symbolic_predict(task, i)
        if sym is not None:
            return sym
        try:
            prompt = encode_prompt(task, test_index=i)
            return decode_prediction(generate(model, prompt))
        except Exception as e:
            print(f"  generation failed for a task ({e}); falling back to a 1x1 placeholder")
            return None
    return predict


SMOKE_TEST = False  # smoke test (version 1, 2026-09-08) completed cleanly in 185.4s with
                    # enable_internet=false: competition data mounted correctly (training/
                    # public-eval/blind-test all present), KV-cache ran without error, and a
                    # correctly-shaped submission.json was produced and validated (all 8
                    # predictions fell back to a placeholder, expected -- 120s of training on
                    # 200 examples produces no valid structured output yet, the same pattern
                    # seen throughout this project at this little training). This run is real.


def main():
    t0 = time.time()
    train_challenges, train_solutions = load_real("arc-agi_training")
    eval_challenges, eval_solutions = load_real("arc-agi_evaluation")
    test_challenges, _ = load_real("arc-agi_test", with_solutions=False)
    print(f"SMOKE_TEST = {SMOKE_TEST}")
    if SMOKE_TEST:
        test_challenges = {k: test_challenges[k] for k in list(test_challenges)[:8]}
        print(f"SMOKE_TEST: blind test subsampled down to {len(test_challenges)} tasks")
    print(f"real data loaded: {len(train_challenges)} training, {len(eval_challenges)} public-eval, "
          f"{len(test_challenges)} blind-test tasks")

    train_examples = build_training_examples(train_challenges, train_solutions, FILTER_CAP)
    print(f"built {len(train_examples)} leave-one-out training examples")
    if SMOKE_TEST:
        train_examples = train_examples[:200]
        print(f"SMOKE_TEST: train_examples subsampled down to {len(train_examples)}")

    train_kwargs = {"max_train_seconds": 120, "eval_every_seconds": 30, "eval_subsample": 5} if SMOKE_TEST else {}
    model = train_model(train_examples, eval_challenges, eval_solutions, **train_kwargs)

    # Sanity-check accuracy on the public evaluation set (known solutions) -- NOT part of the
    # actual submission, just our own confidence check that this checkpoint is reasonable
    # before it becomes the model used for the blind test set below.
    eval_challenges_for_check = eval_challenges
    eval_solutions_for_check = eval_solutions
    if SMOKE_TEST:
        eval_challenges_for_check = {k: eval_challenges[k] for k in list(eval_challenges)[:8]}
        eval_solutions_for_check = {k: eval_solutions[k] for k in eval_challenges_for_check}
    predict_fn = make_verifier_predict_fn(model)
    public_eval_submission = {}
    for task_id, task in eval_challenges_for_check.items():
        entries = []
        for i in range(len(task["test"])):
            pred = predict_fn(task, i)
            attempt = pred if pred is not None else [[0]]
            entries.append({"attempt_1": attempt, "attempt_2": attempt})
        public_eval_submission[task_id] = entries
    public_eval_result = score_submission(public_eval_submission, eval_solutions_for_check)
    print(f"public-eval sanity check: exact_match={public_eval_result['score']:.3f} "
          f"over {len(eval_challenges_for_check)} tasks (not part of the submission)")

    # The actual submission: predict on the blind test set.
    submission = build_submission_template(test_challenges)
    n_symbolic, n_neural, n_fallback = 0, 0, 0
    for task_id, task in test_challenges.items():
        entries = []
        for i in range(len(task["test"])):
            sym = symbolic_predict(task, i)
            if sym is not None:
                pred, source = sym, "symbolic"
            else:
                try:
                    prompt = encode_prompt(task, test_index=i)
                    pred = decode_prediction(generate(model, prompt))
                    source = "neural" if pred is not None else "fallback"
                except Exception as e:
                    print(f"  generation failed for {task_id}[{i}] ({e}); using fallback")
                    pred, source = None, "fallback"
            n_symbolic += source == "symbolic"
            n_neural += source == "neural"
            n_fallback += source == "fallback"
            attempt = pred if pred is not None else [[0]]
            entries.append({"attempt_1": attempt, "attempt_2": attempt})
        submission[task_id] = entries

    problems = validate_submission(submission, test_challenges)
    if problems:
        print("SUBMISSION VALIDATION FAILED:")
        for p in problems[:20]:
            print(" ", p)
        raise RuntimeError(f"{len(problems)} validation problems -- see above")
    print(f"submission.json validated OK: {n_symbolic} symbolic, {n_neural} neural, "
          f"{n_fallback} fallback (of {n_symbolic + n_neural + n_fallback} total test outputs)")

    with open("/kaggle/working/submission.json", "w") as f:
        json.dump(submission, f)
    print(f"\ntotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
