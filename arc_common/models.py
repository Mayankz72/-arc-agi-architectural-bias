"""
Shared model code for the 3 matched-parameter-count backbones (Transformer /
Mamba / hybrid), extracted from notebook/phase2b/phase2b_train.py so Phase 4
(and any future Kaggle script, including the eventual competition submission
notebook) can reuse it instead of re-copy-pasting.

Phase 4 adds incremental (KV-cache) generation on top of the Phase 2b/3
architecture code, unchanged otherwise. Motivation: real ARC-AGI-2 tasks run
2-8x longer than our synthetic benchmark (median real-eval example is ~2,700
tokens vs. ~200-800 for synthetic), and the naive `generate()` used through
Phase 3 recomputes full self-attention over the *entire* growing sequence at
every single generated token -- O(L^2) work per step, O(T*L^2) total. That's
exactly what caused Phase 3's TTA-condition runtime blowup (see PROGRESS.md
2026-09-07 entry), and would be far worse here given real ARC-AGI-2's longer
sequences and larger output grids (up to ~930 tokens vs. ~300 for synthetic).

Design (see PROGRESS.md for the full walkthrough of why this shape):
  - CausalSelfAttentionRoPE / TransformerBlock get a real KV-cache:
    `forward_prefill(x)` processes the whole prompt in one batched call
    (like the old `forward`, but also returns the (k, v) tensors to seed the
    cache), and `forward_step(x, cache, position)` processes exactly one new
    token, attending to the cached keys/values instead of recomputing
    self-attention over the whole sequence. This turns each layer's
    per-generated-token cost from O(current_len^2) into O(current_len) --
    removing the actual quadratic blowup.
  - MambaBlock deliberately does NOT get an incremental step (no
    mamba-ssm InferenceParams wiring): its own forward pass is already
    O(current_len) per call (SSM recurrence, not attention), so redoing it
    from scratch at every generation step costs the same O(T*L) total as
    before -- there was no quadratic problem here to fix. `forward_step`
    just re-runs the full op over the accumulated pre-block sequence (cached
    so it doesn't need to be re-derived from earlier layers) and returns the
    new position's output. This is an intentional scope decision, not an
    oversight: it keeps the risky, error-prone part of this change (an
    actual KV-cache with RoPE) isolated to the one place it's needed.
  - `Backbone.forward` (teacher-forcing, training) is unchanged from
    Phase 2b/3. `Backbone.generate` is rewritten to prefill once, then step
    token-by-token, and produces numerically-equivalent output to the old
    full-recompute `generate` (verified in test_phase4_kvcache.py against a
    non-cached reference forward, on CPU, before this was ever run on
    Kaggle GPUs against the real mamba-ssm kernels).

`mamba_cls` is injectable (defaults to lazily importing `mamba_ssm.Mamba`)
so this module can be unit-tested on a CPU-only machine with a fake
causal substitute op standing in for the real (CUDA-only) Mamba kernel --
see test_phase4_kvcache.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SEQ_LEN = 8192  # generous headroom over real ARC-AGI-2's observed max (~9,306 raw tokens
                     # before any cap is applied; actual training/eval use a lower FILTER_CAP,
                     # see notebook/phase4/*, but RoPE's cache must cover whatever cap is chosen)
ROPE_THETA = 10000.0
DROPOUT = 0.1


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _default_mamba_cls():
    from mamba_ssm import Mamba
    return Mamba


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

    def _qkv(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]  # each [B, n_heads, L, head_dim]

    def _rope(self, q, k, start_pos, length):
        cos = self.cos_cached[start_pos:start_pos + length].to(q.dtype)[None, None, :, :]
        sin = self.sin_cached[start_pos:start_pos + length].to(q.dtype)[None, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q, k

    def forward(self, x):
        """Full non-cached forward over the whole sequence -- used for
        teacher-forced training, unchanged from Phase 2b/3."""
        B, L, D = x.shape
        q, k, v = self._qkv(x)
        q, k = self._rope(q, k, 0, L)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

    def forward_prefill(self, x):
        """Same computation as forward(), but also returns the (k, v) cache
        (post-RoPE-rotation) needed to continue generation incrementally."""
        B, L, D = x.shape
        q, k, v = self._qkv(x)
        q, k = self._rope(q, k, 0, L)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), (k, v)

    def forward_step(self, x, cache, position):
        """x: [B, 1, D], the new token's pre-attention hidden state.
        cache: (k_cache, v_cache) from the previous step, or None on the
        first call after prefill (though in practice this is always
        seeded by forward_prefill before any forward_step call).
        Attends the new query against all cached keys up to and including
        the new position -- O(current_len) instead of recomputing the full
        O(current_len^2) self-attention."""
        q, k, v = self._qkv(x)  # each [B, n_heads, 1, head_dim]
        q, k = self._rope(q, k, position, 1)
        if cache is None:
            k_cache, v_cache = k, v
        else:
            k_cache = torch.cat([cache[0], k], dim=2)
            v_cache = torch.cat([cache[1], v], dim=2)
        # a single new query attending to every key up to & including itself
        # is exactly the causal-masked result for that position -- no mask needed.
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


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, mamba_cls=None):
        super().__init__()
        mamba_cls = mamba_cls or _default_mamba_cls()
        self.ln = nn.LayerNorm(d_model)
        self.mamba = mamba_cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return x + self.drop(self.mamba(self.ln(x)))

    def forward_prefill(self, x):
        """Cache = the pre-block input sequence itself (what forward_step
        needs to concatenate onto and re-run through self.mamba)."""
        out = x + self.drop(self.mamba(self.ln(x)))
        return out, x

    def forward_step(self, x, cache, position):
        """No incremental SSM state (see module docstring) -- re-run the
        full op over the accumulated pre-block sequence and take the last
        position's output. O(current_len) per call, same as the op's own
        cost was already, so no new quadratic blowup, but also no better
        than Phase 2b/3's un-cached generate() for this specific layer type."""
        full_x = x if cache is None else torch.cat([cache, x], dim=1)
        out_full = self.mamba(self.ln(full_x))
        new_out = out_full[:, -1:, :]
        result = x + self.drop(new_out)
        return result, full_x


class Backbone(nn.Module):
    def __init__(self, kind, d_model, n_layers, n_heads=4, d_ff=None, vocab_size=None, mamba_cls=None):
        super().__init__()
        assert vocab_size is not None
        d_ff = d_ff or 4 * d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.use_pos_embed = kind != "transformer"
        if self.use_pos_embed:
            self.pos_embed = nn.Embedding(MAX_SEQ_LEN, d_model)
        if kind == "transformer":
            layers = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        elif kind == "mamba":
            layers = [MambaBlock(d_model, mamba_cls=mamba_cls) for _ in range(n_layers)]
        elif kind == "hybrid":
            layers = []
            for i in range(n_layers):
                if i % 2 == 0:
                    layers.append(MambaBlock(d_model, mamba_cls=mamba_cls))
                else:
                    layers.append(TransformerBlock(d_model, n_heads, d_ff))
        else:
            raise ValueError(kind)
        self.layers = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
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

    def forward_prefill(self, input_ids):
        x = self.embed(input_ids)
        if self.use_pos_embed:
            L = input_ids.size(1)
            positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embed(positions)
        cache = []
        for layer in self.layers:
            x, layer_cache = layer.forward_prefill(x)
            cache.append(layer_cache)
        x = self.ln_f(x)
        return self.head(x), cache

    def forward_step(self, new_token_ids, cache, position):
        """new_token_ids: [B, 1] LongTensor. position: the absolute index
        of this new token in the full sequence (== length of the sequence
        processed so far, since positions are 0-indexed)."""
        x = self.embed(new_token_ids)
        if self.use_pos_embed:
            pos_t = torch.full((x.size(0), 1), position, dtype=torch.long, device=x.device)
            x = x + self.pos_embed(pos_t)
        new_cache = []
        for layer, layer_cache in zip(self.layers, cache):
            x, updated = layer.forward_step(x, layer_cache, position)
            new_cache.append(updated)
        x = self.ln_f(x)
        return self.head(x), new_cache


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def calibrate(vocab_size, d_model=128, n_heads=4, target_kind="transformer", target_layers=4, mamba_cls=None):
    ref = Backbone(target_kind, d_model, target_layers, n_heads, vocab_size=vocab_size, mamba_cls=mamba_cls)
    target = count_params(ref)
    print(f"reference ({target_kind}, {target_layers} layers): {target:,} params")
    configs = {"transformer": target_layers}
    for kind in ["mamba", "hybrid"]:
        best_layers, best_diff = None, float("inf")
        for n_layers in range(2, 25):
            m = Backbone(kind, d_model, n_layers, n_heads, vocab_size=vocab_size, mamba_cls=mamba_cls)
            diff = abs(count_params(m) - target)
            if diff < best_diff:
                best_diff, best_layers = diff, n_layers
            del m
        configs[kind] = best_layers
        m = Backbone(kind, d_model, best_layers, n_heads, vocab_size=vocab_size, mamba_cls=mamba_cls)
        print(f"{kind}: {best_layers} layers -> {count_params(m):,} params (target {target:,})")
    return configs


@torch.no_grad()
def generate(model, prompt_tokens, eos_id, max_new_tokens, device):
    """Prefill the prompt once, then step token-by-token using each layer's
    cache. Numerically equivalent to the old full-recompute generate() (see
    test_phase4_kvcache.py), but each step after prefill costs O(current_len)
    instead of O(current_len^2) for every attention layer."""
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    logits, cache = model.forward_prefill(ids)
    next_id = int(logits[0, -1].argmax())
    generated = [next_id]
    position = ids.size(1)
    if next_id != eos_id:
        for _ in range(max_new_tokens - 1):
            new_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, cache = model.forward_step(new_tok, cache, position)
            next_id = int(logits[0, -1].argmax())
            generated.append(next_id)
            position += 1
            if next_id == eos_id:
                break
    return generated
