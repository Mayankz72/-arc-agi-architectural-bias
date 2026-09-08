"""
Debug why the matched-parameter Transformer produced zero valid structured
output in the Phase 2 pilot (see ROADMAP.md) while Mamba/hybrid partially
learned. Tests the standard "transformers need LR warmup + grad clipping"
hypothesis by training ONLY the transformer under a few configs, faster than
a full 3-architecture run since Mamba/hybrid are skipped entirely.
"""
import json
import subprocess
import sys
import time
import copy

print("Installing mamba-ssm (not used here, but kept for parity with phase2_train.py's environment)...")
# not needed for this debug script (transformer only) -- skip install to save time

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
from arc_common.scoring import score_submission  # noqa: E402

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


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):
        L = x.size(1)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class TransformerBackbone(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.embed = nn.Embedding(FULL_VOCAB, d_model)
        self.pos_embed = nn.Embedding(MAX_SEQ_LEN, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
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


def train_config(name, train_examples, eval_challenges, eval_solutions, d_model=128, n_layers=4, n_heads=4,
                  epochs=40, batch_size=32, lr=3e-4, warmup_frac=0.0, grad_clip=None, weight_decay=1e-2,
                  eval_every=10, eval_subsample=30):
    torch.manual_seed(0)
    model = TransformerBackbone(d_model, n_layers, n_heads).to(DEVICE)
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

    print(f"\n=== config '{name}': lr={lr} warmup_frac={warmup_frac} grad_clip={grad_clip} "
          f"({count_params(model):,} params, {total_steps} total steps) ===")

    best_score, best_state, best_epoch = -1.0, None, -1
    step = 0
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
            step += 1
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
        dict(name="baseline (no warmup, no clip, lr=3e-4)", lr=3e-4, warmup_frac=0.0, grad_clip=None),
        dict(name="warmup+clip, lr=3e-4", lr=3e-4, warmup_frac=0.05, grad_clip=1.0),
        dict(name="warmup+clip, lr=1e-3", lr=1e-3, warmup_frac=0.05, grad_clip=1.0),
    ]

    results = {}
    for cfg in configs:
        name = cfg.pop("name")
        _, score = train_config(name, train_examples, train_eval_challenges, train_eval_solutions, **cfg)
        results[name] = score

    print("\n=== SUMMARY ===")
    for name, score in results.items():
        print(f"{name}: best quickcheck_cell_acc = {score:.3f}")

    with open("/kaggle/working/transformer_debug_results.json", "w") as f:
        json.dump(results, f)
    print(f"\ntotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
