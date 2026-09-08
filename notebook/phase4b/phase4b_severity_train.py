"""
Phase 4b: compositional-split-severity sweep on the synthetic benchmark --
the "{3 architectures} x {compositional-split severity}" half of the original
Phase 4 plan (ROADMAP.md section 5, Phase 4). Model/training code is copied
verbatim from notebook/phase2b/phase2b_train.py (same RoPE-fixed Transformer,
same shared training regime validated across Phase 2b/2c) -- this experiment
only varies the DATA, not the architectures or training procedure, so reusing
proven code exactly is the right call (unlike Phase 4a's real-data run, these
sequences are short and the Phase 2b/3 code already runs fast at this scale --
no need for the newer KV-cache).

Severity levels (see generator.py's `split_for_severity`, added 2026-09-07):
the 4x4 (structural x color) combo grid decomposes into 4 disjoint "diagonals"
(mod-4 shifts of the pairing). Holding out `severity` of them (1, 2, or 3)
leaves fewer training combos and more held-out ones, while preserving the
original design's key property at every level: every individual primitive
still appears in at least one training combo, so held-out failure stays about
novel RECOMBINATION, not about primitives never seen at all. severity=1 is
exactly Phase 2b/3's original split (12 train / 4 heldout combos); severity=2
and 3 are new (8/8 and 4/12).

Data is generated ON THE FLY inside this kernel by calling `generate_split`
from generator.py (already in the shared Kaggle Dataset, unchanged) with
combo lists computed by `diagonal`/`split_for_severity`, inlined below --
this avoids needing an updated Dataset (the storage.googleapis.com upload
block from Phase 3/4a is still unresolved) for the two new severity levels.
Training task COUNT is held fixed at 600 total across severities
(tasks_per_train_combo = 600 // n_train_combos) so severity isolates
compositional difficulty from total training-data volume, not data quantity.
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
from generator import generate_split, STRUCT_NAMES, COLOR_NAMES, ALL_COMBOS  # noqa: E402
from mamba_ssm import Mamba  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

PAD = VOCAB_SIZE
FULL_VOCAB = VOCAB_SIZE + 1

# --- Phase 4b severity helper (inlined; canonical version lives in generator.py,
# kept in sync by hand since the shared Dataset's generator.py copy predates this) ---

def diagonal(k):
    return [(STRUCT_NAMES[i], COLOR_NAMES[(i + k) % 4]) for i in range(4)]


def split_for_severity(severity):
    assert 1 <= severity <= 3
    heldout = [combo for k in range(severity) for combo in diagonal(k)]
    train = [combo for combo in ALL_COMBOS if combo not in heldout]
    return train, heldout


# --- data ---------------------------------------------------------------------

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
    return configs


# --- training / eval (verbatim from phase2b_train.py) ------------------------

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
            print(f"  epoch {epoch}: loss {total_loss / n_batches:.4f}  quickcheck_cell_acc {score:.3f}")
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
def evaluate(model, challenges, solutions, kind_label, split_label):
    model.eval()
    submission = {}
    cell_accs, shape_matches = [], 0
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
    all_results = {}
    for severity in [1, 2, 3]:
        train_combos, heldout_combos = split_for_severity(severity)
        tasks_per_train_combo = 600 // len(train_combos)
        print(f"\n{'=' * 80}\nSEVERITY {severity}: {len(train_combos)} train combos, "
              f"{len(heldout_combos)} heldout combos, {tasks_per_train_combo} tasks/train-combo\n{'=' * 80}")

        train_split = generate_split(train_combos, n_tasks_per_combo=tasks_per_train_combo, seed=0)
        train_eval_split = generate_split(train_combos, n_tasks_per_combo=10, seed=2)
        heldout_split = generate_split(heldout_combos, n_tasks_per_combo=50, seed=1)

        train_challenges, train_solutions = train_split["challenges"], train_split["solutions"]
        train_eval_challenges, train_eval_solutions = train_eval_split["challenges"], train_eval_split["solutions"]
        heldout_challenges, heldout_solutions = heldout_split["challenges"], heldout_split["solutions"]

        train_examples = build_training_examples(train_challenges, train_solutions)
        print(f"built {len(train_examples)} leave-one-out training examples")

        configs = calibrate()

        severity_results = {}
        for kind in ["transformer", "mamba", "hybrid"]:
            model = train_model(kind, configs[kind], train_examples, train_eval_challenges, train_eval_solutions)
            severity_results[kind] = {
                "n_params": count_params(model),
                "train_eval": evaluate(model, train_eval_challenges, train_eval_solutions, kind, f"sev{severity} train_eval"),
                "heldout": evaluate(model, heldout_challenges, heldout_solutions, kind, f"sev{severity} heldout"),
            }
        all_results[f"severity_{severity}"] = {
            "n_train_combos": len(train_combos), "n_heldout_combos": len(heldout_combos),
            "architectures": severity_results,
        }

    print("\n=== SUMMARY (compositional-split-severity sweep) ===")
    for sev_key, sev_data in all_results.items():
        print(f"\n{sev_key} ({sev_data['n_train_combos']} train / {sev_data['n_heldout_combos']} heldout combos):")
        for kind, r in sev_data["architectures"].items():
            te, ho = r["train_eval"], r["heldout"]
            print(f"  {kind}: params={r['n_params']:,} "
                  f"train_eval(cell={te['mean_cell_acc']:.3f} shape={te['shape_match_rate']:.3f}) "
                  f"heldout(cell={ho['mean_cell_acc']:.3f} shape={ho['shape_match_rate']:.3f})")

    def strip(d):
        return {k: v for k, v in d.items() if k != "per_task"}

    out = {
        sev_key: {
            "n_train_combos": sev_data["n_train_combos"],
            "n_heldout_combos": sev_data["n_heldout_combos"],
            "architectures": {
                kind: {"n_params": r["n_params"], "train_eval": strip(r["train_eval"]), "heldout": strip(r["heldout"])}
                for kind, r in sev_data["architectures"].items()
            },
        }
        for sev_key, sev_data in all_results.items()
    }
    with open("/kaggle/working/phase4b_results.json", "w") as f:
        json.dump(out, f)
    print(f"\ntotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
