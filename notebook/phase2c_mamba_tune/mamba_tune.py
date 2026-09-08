"""
Phase 2b's comparison (see ROADMAP.md) used one training regime -- lr=1e-3,
warmup_frac=0.05, grad_clip=1.0, weight_decay=1e-2 -- for all three
architectures, even though it was tuned specifically to fix the Transformer's
positional-embedding bug. That's flagged as an open caveat: Mamba's weak
scores (cell-acc 0.045-0.075 across 2 seeds, vs. the Transformer's 0.19-0.31)
might partly reflect a regime that's suboptimal for an SSM, not a real
ceiling on what Mamba can do on this task.

This script trains ONLY Mamba (skips Transformer/hybrid to save time) under 3
regimes, matched in every other respect (same params, same 80-epoch budget,
same data) to Phase 2b's Mamba config, to test that caveat directly:
  1. current (phase2b's regime): lr=1e-3, warmup=0.05, clip=1.0, wd=1e-2
  2. original-pilot regime (what worked OK for Mamba before the Transformer
     fix existed): lr=3e-4, no warmup, no clip, wd=1e-2
  3. no weight decay: same as (1) but wd=0.0, testing whether weight decay
     specifically hurts Mamba's recurrent state-space parameters
"""
import json
import subprocess
import sys
import time
import copy

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
    raise RuntimeError("could not locate arc_common under /kaggle/input")
print("Using DATASET_ROOT:", DATASET_ROOT)
sys.path.insert(0, DATASET_ROOT)
sys.path.insert(0, f"{DATASET_ROOT}/vendor/arc-dsl")

from arc_common.tokenizer import (  # noqa: E402
    EOS, VOCAB_SIZE, encode_prompt, encode_target, decode_prediction,
)
from mamba_ssm import Mamba  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

PAD = VOCAB_SIZE
FULL_VOCAB = VOCAB_SIZE + 1
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


DROPOUT = 0.1
MAX_SEQ_LEN = 2048


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return x + self.drop(self.mamba(self.ln(x)))


class MambaOnlyBackbone(nn.Module):
    """Matches phase2b's Backbone(kind="mamba") exactly: additive learned
    positional embedding + a stack of MambaBlocks. n_layers=5 to match
    phase2b's calibrated param count (848,256) for direct comparability."""

    def __init__(self, d_model=128, n_layers=5):
        super().__init__()
        self.embed = nn.Embedding(FULL_VOCAB, d_model)
        self.pos_embed = nn.Embedding(MAX_SEQ_LEN, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, FULL_VOCAB, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids):
        L = input_ids.size(1)
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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


def train_config(name, train_examples, eval_challenges, eval_solutions, d_model=128, n_layers=5,
                  epochs=80, batch_size=32, lr=1e-3, warmup_frac=0.05, grad_clip=1.0, weight_decay=1e-2,
                  eval_every=20, eval_subsample=30):
    torch.manual_seed(0)
    model = MambaOnlyBackbone(d_model, n_layers).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(train_examples)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda) if warmup_frac > 0 else None

    sample_ids = list(eval_challenges)[:eval_subsample]
    sub_challenges = {k: eval_challenges[k] for k in sample_ids}
    sub_solutions = {k: eval_solutions[k] for k in sample_ids}

    print(f"\n=== config '{name}': lr={lr} warmup_frac={warmup_frac} grad_clip={grad_clip} wd={weight_decay} "
          f"({count_params(model):,} params, {total_steps} total steps) ===")

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
            if sched is not None:
                sched.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % eval_every == 0 or epoch == epochs - 1:
            score = quick_cell_acc(model, sub_challenges, sub_solutions)
            cur_lr = opt.param_groups[0]["lr"]
            print(f"  epoch {epoch}: loss {total_loss / n_batches:.4f}  lr {cur_lr:.2e}  quickcheck_cell_acc {score:.3f}")
            if score >= best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
    print(f"  '{name}' best checkpoint: epoch {best_epoch} (quickcheck_cell_acc {best_score:.3f})")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def main():
    t0 = time.time()
    train_challenges, train_solutions = load_split("train")
    train_eval_challenges, train_eval_solutions = load_split("train_eval")
    train_examples = build_training_examples(train_challenges, train_solutions)
    print(f"built {len(train_examples)} leave-one-out training examples")

    configs = [
        dict(name="current (phase2b regime): lr=1e-3 warmup=0.05 clip=1.0 wd=1e-2",
             lr=1e-3, warmup_frac=0.05, grad_clip=1.0, weight_decay=1e-2),
        dict(name="original-pilot regime: lr=3e-4 no-warmup no-clip wd=1e-2",
             lr=3e-4, warmup_frac=0.0, grad_clip=None, weight_decay=1e-2),
        dict(name="no weight decay: lr=1e-3 warmup=0.05 clip=1.0 wd=0.0",
             lr=1e-3, warmup_frac=0.05, grad_clip=1.0, weight_decay=0.0),
    ]

    results = {}
    for cfg in configs:
        name = cfg.pop("name")
        _, score = train_config(name, train_examples, train_eval_challenges, train_eval_solutions, **cfg)
        results[name] = score

    print("\n=== SUMMARY ===")
    for name, score in results.items():
        print(f"{name}: best quickcheck_cell_acc = {score:.3f}")

    with open("/kaggle/working/mamba_tune_results.json", "w") as f:
        json.dump(results, f)
    print(f"\ntotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
