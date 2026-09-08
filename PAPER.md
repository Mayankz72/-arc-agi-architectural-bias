# Positional Inductive Bias, Not Parameter Count, Drives Compositional Generalization on ARC-Style Grid Reasoning

**Author:** Mayank Mishra
**Track:** ARC Prize 2026 Paper Track
**Code:** [repository link] · **Kaggle submission:** `kojijhjughio/arc-agi-2-submission`

## Abstract

We ask whether backbone architecture — Transformer, Mamba/SSM, or a Transformer+Mamba hybrid — changes how well a model trained *from scratch* generalizes to novel compositions of grid-transformation rules it never saw during training. Following the controlled synthetic-manipulation methodology of Kallini et al. (2024), we build a systematic compositional split of ARC-style grid transformations (4 structural primitives × 4 color primitives, with a tunable fraction of primitive *pairings* held out entirely from training) and train three matched-parameter-count (~0.8–1.1M) backbones from scratch under an identical regime. We find that a rotary-position-encoded (RoPE) Transformer generalizes to unseen (structural, color) pairings 4–6× better than parameter-matched Mamba and hybrid backbones, and that this gap **widens monotonically and dramatically as compositional severity increases** — from a 19% relative train/held-out accuracy drop when 1 of 4 primitive pairings is held out, to 87% when 3 of 4 are held out. Mamba and hybrid backbones never learn their own training distribution well enough at this scale for the same comparison to be interpretable. We attach an NSA-style (arXiv:2501.04424) symbolic proposer+verifier layer and find it provably solves our synthetic benchmark by construction (an expected ceiling effect, not an architecture finding) but rarely fires on real ARC-AGI-2 tasks, whose transformation space is not exhaustively enumerable by a search this small. On real ARC-AGI-2 data, the Transformer's advantage replicates directionally but at a much smaller margin (~1.7×), and a from-scratch ~800K-parameter Transformer submitted to the competition scores 0.00 exact-match — expected for a benchmark deliberately designed to defeat small, non-test-time-trained models, and consistent with our own non-exact partial-credit metrics showing real, if shallow, learning. Our contribution is a controlled demonstration that *architecture's positional inductive bias*, not scale, is what determines whether a model can recombine previously-seen primitives into novel combinations — and a full account, including negative results, of what it takes to measure that cleanly.

## 1. Introduction

Frontier LLMs now score well on ARC-AGI-2 with internet access and unconstrained compute, but the Kaggle-Systems category this competition actually judges — a single notebook, no internet, ≤12h runtime — has historically topped out far lower. Within that regime, small models trained from scratch on synthetic and real ARC-style data are the realistic unit of study, and the open question we address is architectural: for models that must recombine familiar primitives into novel combinations at test time, does the choice of sequence-mixing operator (self-attention vs. state-space recurrence) matter, and if so, why?

No prior ARC-AGI work directly isolates this variable. Prior architecture comparisons on ARC either differ in training data and objective simultaneously (making the architecture's contribution unclear) or study only one backbone family. We instead adapt the controlled-manipulation methodology of *Mission: Impossible Language Models* (Kallini et al., ACL 2024): rather than comparing architectures on a fixed, uncontrolled benchmark, we build the benchmark ourselves so that the *only* thing separating "seen" from "novel" is a well-defined recombination of primitives the model has already been trained on individually. This lets us attribute a generalization gap to a specific mechanism rather than to an uncontrolled confound.

## 2. Prior Work

| Approach | Mechanism | Relation to this work |
|---|---|---|
| Icecuber (2020 ARC winner) | Hand-crafted 142-function DSL, greedy DAG search | Reference point for a pure-search ceiling (~20%); doesn't generalize past hand-enumerated primitives |
| Akyürek et al., Test-Time Training (arXiv:2411.07279) | Per-task fine-tuning at inference via self-generated augmentations | Orthogonal axis (inference-time adaptation) to the architecture question we study; not used here by design |
| MindsAI (ARC Prize 2024 top score, 55.5%) | T5 backbone + synthetic pretraining + TTT | Not reproducible (not open-sourced); not an architecture-controlled comparison |
| ARChitects, NVARC (2024 ARC Prize top solutions) | Prior-year top entries in the same no-internet, ≤12h "Kaggle Systems" compute regime this competition judges; NVARC's category topped out around 27.6% | Establish that our compute/runtime regime is realistic for competitive entries even without internet-scale pretraining; neither isolates architecture as a variable |
| NSA: Neuro-Symbolic ARC Challenge (arXiv:2501.04424) | Transformer proposer (top-k candidate primitives) + bounded DSL search, verified against every demonstration pair; also uses substantial test-time fine-tuning | We reuse the proposer+verifier pattern (Section 4.3) but deliberately omit its test-time fine-tuning, to keep our ablation about architecture, not adaptation |
| "ARC Is a Vision Problem!" / VARC (arXiv:2511.14761) | ViT over a rendered canvas, 2D patch embedding, heavy rotation/reflection/scale test-time-training augmentation | Motivates our TTA experiment (Section 4.3) and explains, in hindsight, why our own un-augmented TTA fails |
| "ARC-AGI Without Pretraining" / CompressARC (arXiv:2512.06104) | 76K-parameter equivariant network, minimum-description-length optimization at inference, no pretraining at all | Validates from-scratch training as a legitimate, precedented regime; a complementary point in the design space (representation/objective) rather than architecture |
| Mission: Impossible Language Models (Kallini et al., ACL 2024) | Controlled synthetic-language manipulation; GPT-2 learns linguistically "possible" languages faster than matched "impossible" ones because of an information-locality inductive bias, not innate grammar | **Methodology template**: controlled construction of the training distribution plus multiple converging metrics, so a learning-speed or generalization difference can be attributed to a specific inductive bias rather than reported as a bare score gap |

## 3. Approach

### 3.1 Synthetic compositional benchmark

We define 4 structural primitives (`rot90`, `hmirror`, `vmirror`, `upscale2`) and 4 color primitives (`swap` on 4 disjoint color pairs), each drawn from Michael Hodel's `arc-dsl` (MIT licensed). Every task applies one structural primitive then one color primitive to a random small grid, giving 16 possible (structural, color) pairings. We hold out a controllable subset of these pairings entirely from training — never demonstrated in any training task — while guaranteeing every individual primitive still appears in at least one training pairing. This isolates novel *recombination* as the generalization target, distinct from novel primitives.

We parameterize **compositional severity** as the number of held-out pairings, organized as disjoint "diagonals" of the 4×4 pairing grid (mod-4 shifts): severity 1 holds out 4 of 16 pairings (12 train / 4 held-out), severity 2 holds out 8 (8/8), severity 3 holds out 12 (4/12). Total training task count is held fixed at 600 across severities, so severity isolates compositional difficulty from training-data volume.

### 3.2 Matched-parameter architectures

We compare three backbones, each ~0.8–1.1M parameters, sharing a token embedding, weight-tied LM head, and (for Mamba/hybrid) a learned additive positional embedding:

- **Transformer**: decoder-only self-attention with rotary positional embeddings (RoPE), no additive positional embedding.
- **Mamba**: a stack of Mamba/SSM blocks (`mamba-ssm`).
- **Hybrid**: Mamba and Transformer blocks interleaved 1:1; Transformer sub-blocks use RoPE, Mamba sub-blocks keep the additive embedding.

Grids are tokenized as a flattened row-major sequence (10 color tokens + 6 structural tokens: row-separator, grid-in/out markers, pair-end, BOS, EOS). Training uses leave-one-out augmentation (each demonstration pair in a task is held out once as the target, with the remaining pairs as context), AdamW with a shared, independently-validated hyperparameter regime (lr 1e-3, linear warmup, gradient clipping, weight decay), and checkpoint selection against a held-in (never held-out) validation subsample.

### 3.3 Neuro-symbolic verifier

Following NSA's proposer+verifier pattern, we add a symbolic layer that searches a bounded space (14 structural operations × ~136 color operations from `arc-dsl`) for a program consistent with *every* demonstration pair in a task, and applies it to the test input when found. Unlike NSA, we do not use test-time fine-tuning, keeping this an architecture-agnostic filter rather than an adaptation mechanism. We separately implement D4 (8-view rotation/reflection) test-time augmentation with majority voting, a standard cheap technique from prior Kaggle ARC solutions.

### 3.4 Real ARC-AGI-2 evaluation and submission

For real-data evaluation and the competition submission, autoregressive generation over long token sequences becomes a genuine bottleneck: naive re-computation of self-attention over the whole sequence at every generated token is O(length²) per step. We implement a standard KV-cache (`forward_prefill`/`forward_step`) for the RoPE attention path, verified bit-exact against a non-cached reference on CPU before any GPU use, reducing this to O(length) per step. Mamba's own per-step cost was already linear and is left unchanged. The competition submission trains a Transformer-only model (no Mamba/hybrid, both because it is the strongest architecture in every experiment reported here and because `mamba-ssm` requires a `pip install` that a real, no-internet scored run cannot perform) from scratch on the real ARC-AGI-2 training set within a wall-clock training budget, then predicts on the blind test set via the symbolic verifier with a KV-cached neural fallback.

## 4. Results

### 4.1 Architecture comparison on the synthetic benchmark

At severity 1 (the base split), across two independent seeds:

| Model | Params | train_eval cell-acc | held-out cell-acc | held-out shape-match |
|---|---|---|---|---|
| **Transformer (RoPE)** | 0.80M | 0.309 / 0.275 | **0.235 / 0.191** | **0.380 / 0.325** |
| Mamba | 0.85M | 0.045 / 0.075 | 0.034 / 0.032 | 0.070 / — |
| Hybrid | 0.89M | 0.050 / 0.046 | 0.069 / 0.045 | 0.105 / — |

*(pairs are seed 0 / seed 1)*

The RoPE-Transformer outperforms Mamba and the hybrid by 4–6× on held-out compositions in both seeds. This required first diagnosing and fixing an unrelated failure mode: a vanilla Transformer with a learned absolute positional embedding produced *zero* structurally valid output at any training budget (0.000 across the board), a failure isolated via ablation to the positional scheme specifically (not exposure bias, not learning rate) and resolved by switching to RoPE — after which the Transformer reversed from worst to best. A subsequent hyperparameter sweep confirmed Mamba's weak scores are not an artifact of a Transformer-tuned shared training regime (its own regime performed worse).

### 4.2 The compositional generalization gap widens with severity

This is the paper's central mechanistic result:

| Severity (train/held-out pairings) | Transformer train_eval cell-acc | Transformer held-out cell-acc | Relative drop |
|---|---|---|---|
| 1 (12/4) | 0.298 | 0.241 | 19.1% |
| 2 (8/8) | 0.377 | 0.156 | 58.6% |
| 3 (4/12) | **0.852** | **0.108** | **87.3%** |

As severity increases — fewer training pairings, each seen with proportionally more data since total training volume is held fixed — the Transformer's in-distribution accuracy climbs sharply (fitting 4 well-drilled pairings is easier than fitting 12 sparser ones) while its held-out accuracy on genuinely novel pairings *falls*. Shape-match tells the identical story (19.2% → 57.9% → 76.0% relative drop). Mamba and hybrid show the same directional trend, but their own train_eval scores at severity 3 (0.124, 0.125) are far below the Transformer's (0.852), meaning neither ever learns its own training distribution well enough at this scale/budget for a "failed to generalize" vs. "failed to learn" distinction to be meaningful for them.

### 4.3 The symbolic verifier hits its predicted ceiling; naive TTA hurts

Attaching the symbolic verifier gives a perfect 1.000 exact-match on both synthetic splits for all three architectures — expected, since the bounded search space is a strict superset of the generator's own primitives, and reported here as a pipeline-correctness check rather than an architecture finding. Naive D4 test-time augmentation, by contrast, is a **negative result**: it reduced held-out cell-accuracy for the Transformer (0.199 → 0.126, −37% relative) and collapsed Mamba's to exactly 0.000. None of the three backbones were trained with rotation/reflection augmentation, so they have no equivariance for majority-voting across symmetric views to exploit — a concrete illustration of why VARC's own use of heavy training-time augmentation alongside its TTA is necessary, not incidental.

### 4.4 Real ARC-AGI-2: the finding transfers, at a smaller margin

An initial real-data comparison (8 epochs) showed the Transformer scoring a flat 0.000 while Mamba/hybrid scored small nonzero values — reading, at first glance, like a reversal. The shape-match column revealed the actual cause: the Transformer produced *zero* structurally valid output at all (matching the pre-RoPE collapse signature from Section 4.1), an under-training artifact rather than a real effect. A follow-up at 40 epochs confirmed this: the Transformer's quickcheck score stayed at 0.000 through epoch 5 (the same dead zone) before climbing to a peak at epoch 35, mirroring the synthetic Transformer's own slow start (which took 60 of 80 epochs to peak). A matched-epoch (40 epochs, all three architectures) final comparison:

| Model | Params | cell-acc | shape-match |
|---|---|---|---|
| **Transformer (RoPE)** | 795,520 | **0.079** | **0.108** |
| Mamba | 1,022,336 | 0.047 | 0.072 |
| Hybrid | 1,103,872 | 0.044 | 0.065 |

The Transformer's advantage transfers directionally to real, far more diverse data, at roughly 1.7× rather than 4–6×. We do not yet distinguish whether this smaller margin reflects real ARC-AGI-2's diversity leaving less room for RoPE's specific fix to dominate, or whether even 40 epochs is still short of each architecture's actual ceiling.

### 4.5 Competition submission

A from-scratch ~800K-parameter Transformer, trained on the real ARC-AGI-2 training set for a 6-hour wall-clock budget (90 epochs) and evaluated with the symbolic-verifier-plus-KV-cached-neural pipeline, scores **0.00 exact-match** on the competition's held-out test set. This is consistent with our own pre-submission public-evaluation check (also 0.000) and expected given ARC-AGI-2 is explicitly constructed to resist small, non-test-time-trained models: comparable published approaches without test-time training or a vision-native representation similarly fall to near-zero on ARC-AGI-2 even at far larger scale (e.g., VARC's own 18M-parameter model drops from 54–60% on ARC-AGI-1 to 8–11% on ARC-AGI-2). Our own non-exact metrics (Section 4.4) confirm the model is not learning nothing — it recovers correct grid dimensions on roughly 1 in 9 real tasks and individual cells correctly about 8% of the time on average — but exact whole-grid correctness on a genuinely novel task is a far higher bar that a model this small, with no test-time adaptation, is unlikely to clear.

## 5. Conclusion

Holding representation, training regime, verifier, and parameter count fixed, we isolated a single variable — the sequence-mixing operator's inductive bias — and found it determines whether a from-scratch model can recombine previously-seen primitives into genuinely novel combinations. A RoPE-equipped Transformer does this far better than parameter-matched Mamba and hybrid backbones on a controlled synthetic benchmark, and the gap between fitting a training distribution and generalizing beyond it widens dramatically and monotonically as the compositional demand increases — a direct, mechanistic account of *why* architecture matters here, not merely *that* it does. The same direction of effect, at a smaller margin, replicates on real ARC-AGI-2 data. Two honest limitations bound these claims: this is one task family and one small parameter scale, and our real-data comparison used a training budget that had not yet clearly plateaued for any of the three architectures. We also report, without overclaiming, that a small enumerable-DSL symbolic verifier and naive un-augmented test-time augmentation both behave exactly as their own preconditions predict — solving everything inside their search space and nothing outside it, respectively — which we see as a useful negative-result contribution in its own right for anyone assembling a similar neuro-symbolic pipeline.

## Reproducibility

All code, data generators, trained-model configurations, and full experimental logs are available at [repository link], including the local unit tests (`test_harness.py`, `test_phase3_harness.py`, `test_phase4_kvcache.py`) that verify the tokenizer, scoring harness, symbolic verifier, and KV-cache against reference implementations before any Kaggle GPU time was spent. The Kaggle submission notebook (`kojijhjughio/arc-agi-2-submission`) reproduces Section 4.5 end-to-end with no internet access.
