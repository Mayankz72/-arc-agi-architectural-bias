"""
Correctness test for arc_common.models' KV-cache (forward_prefill +
forward_step) against the plain non-cached forward(), run entirely on CPU.

Real mamba-ssm needs a CUDA kernel and can't run on this dev machine, so
MambaBlock is tested with a fake causal substitute op (`FakeCausalOp`, same
constructor signature as mamba_ssm.Mamba) standing in for the real kernel.
This validates the *caching/accumulation logic* in MambaBlock.forward_step
(concatenate onto the cached pre-block sequence, rerun the op, take the last
position) independently of what the op itself computes -- the logic only
requires the op be a causal sequence-to-sequence function, which is true of
both the fake op here and the real Mamba kernel. The Transformer/RoPE path
uses the real attention math (no substitute needed) since it doesn't depend
on any CUDA-only kernel.

If this test passes, the KV-cache is mathematically exact, not just "runs
without crashing" -- run this BEFORE ever trusting the cache on a real
Kaggle GPU run.
"""
import torch
import torch.nn as nn

from arc_common.models import Backbone, generate

torch.manual_seed(0)

VOCAB_SIZE = 20
EOS_ID = 19


class FakeCausalOp(nn.Module):
    """Stand-in for mamba_ssm.Mamba: same constructor signature, and a
    genuinely causal (output at position i depends only on inputs <= i)
    sequence-to-sequence function, so it exercises MambaBlock's
    forward_step accumulation logic the same way the real kernel would."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mix = nn.Linear(d_model, d_model)

    def forward(self, x):
        # causal cumulative mean, then a learned mix -- deliberately not the
        # identity or anything attention-like, just something that genuinely
        # depends on the whole causal prefix so a caching bug would show up.
        cumsum = torch.cumsum(x, dim=1)
        counts = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        causal_mean = cumsum / counts
        return self.mix(causal_mean)


@torch.no_grad()
def reference_generate(model, prompt_tokens, eos_id, max_new_tokens, device):
    """The old (Phase 2b/3) full-recompute generate(), kept here only as the
    ground truth for this test."""
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(ids)
        next_id = int(logits[0, -1].argmax())
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos_id:
            break
    return ids[0, len(prompt_tokens):].tolist()


def check_kind(kind, mamba_cls=None):
    model = Backbone(kind, d_model=32, n_layers=4, n_heads=4, vocab_size=VOCAB_SIZE, mamba_cls=mamba_cls)
    model.eval()
    device = "cpu"

    for trial in range(5):
        prompt_len = torch.randint(3, 12, (1,)).item()
        prompt = torch.randint(0, VOCAB_SIZE - 1, (prompt_len,)).tolist()  # avoid EOS in the prompt itself

        ref = reference_generate(model, prompt, EOS_ID, max_new_tokens=15, device=device)
        cached = generate(model, prompt, EOS_ID, max_new_tokens=15, device=device)

        assert ref == cached, (
            f"[{kind}] trial {trial}: cached generate() diverged from the non-cached reference.\n"
            f"prompt={prompt}\nreference={ref}\ncached={cached}"
        )

        # Also check raw logits agreement at every step (not just the argmax
        # sequence, which a bug could get right by coincidence on some steps)
        full_seq = prompt + ref
        ref_logits = model(torch.tensor([full_seq]))[0]  # [L, vocab] teacher-forced over the true realized sequence
        logits0, cache = model.forward_prefill(torch.tensor([prompt]))
        step_logits = [logits0[0, -1]]
        position = len(prompt)
        for tok in ref[:-1] if len(ref) > 1 else []:
            step_out, cache = model.forward_step(torch.tensor([[tok]]), cache, position)
            step_logits.append(step_out[0, -1])
            position += 1
        for i, sl in enumerate(step_logits):
            rl = ref_logits[len(prompt) - 1 + i]
            assert torch.allclose(sl, rl, atol=1e-4), (
                f"[{kind}] trial {trial} step {i}: logits mismatch (max diff {(sl - rl).abs().max().item()})"
            )
    print(f"[{kind}] KV-cache matches non-cached reference exactly across 5 random trials (sequences + logits)")


if __name__ == "__main__":
    check_kind("transformer")
    check_kind("mamba", mamba_cls=FakeCausalOp)
    check_kind("hybrid", mamba_cls=FakeCausalOp)
    print("ALL OK")
