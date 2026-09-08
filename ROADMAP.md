# Architectural Inductive Bias on ARC-AGI-2

**Target:** Kaggle ARC Prize 2026 (ARC-AGI-2 track) + ARC Prize Paper Track
**Author:** Mayank Mishra
**Status (as of 2026-09-08):** Phases 0–5 complete (synthetic benchmark, 3-architecture comparison, neuro-symbolic verifier, real-ARC-AGI-2 evaluation, working Kaggle submission), repo open-sourced, `PAPER.md` drafted, first competition submission made and scored (0.00, expected — see Phase 6). Remaining: proofread/submit the paper to the actual Paper Track form, lock in the Final Submission selection near the Oct 25–31 window, a handful of explicitly-optional follow-up experiments (see `PROGRESS.md`'s Next Up list).

---

## 1. The research question

Does backbone architecture (Transformer vs. Mamba/SSM vs. a Transformer+Mamba hybrid) change how well a model trained *from scratch* generalizes to novel compositions of grid-transformation rules it never saw during training?

This applies the controlled-experiment methodology of **Kallini et al., "Mission: Impossible Language Models" (ACL 2024 Best Paper)** — train small models from scratch on systematically manipulated synthetic data, measure what generalizes — to a new domain: ARC-style abstract visual reasoning. As of this writing, no public paper directly tests architectural inductive bias for compositional generalization on ARC-AGI-style tasks. That gap is the novel contribution.

This is deliberately **not** an attempt to chase the highest leaderboard score. The Paper Track rubric weighs Theory, Universality, and Novelty equally with Accuracy — the organizers are explicitly asking for understanding, not just a bigger number.

## 2. Why this is tractable (not a moonshot)

- Frontier LLMs (GPT-6 Astra, Claude Opus 5, etc.) now score 85%+ on ARC-AGI-2 — but only with internet access, proprietary weights, and no compute constraints. That's a different competition category ("Base LLMs" / "Reasoning Systems" on the public leaderboard).
- The category this competition actually judges — **Kaggle Systems**: Kaggle Notebook only, no internet, ≤12h runtime, open-source weights — tops out historically around **27.6%** (NVARC, prior year). Small-from-scratch models trained on synthetic + real ARC tasks are competitive within that regime; nobody expects 85% here.
- Training three small (matched-parameter) models from scratch on synthetic + ARC-scale data is realistic on a single GPU (or Kaggle/Colab free tier) — no need for TTT-style per-task fine-tuning of a giant pretrained LM.

## 3. Competition facts (verified live, 2026-09-06)

**ARC Prize 2026 – ARC-AGI-2** (kaggle.com/competitions/arc-prize-2026-arc-agi-2) — a relaunch of ARC Prize 2025, so 2024/2025 writeups (MindsAI, ARChitects, NVARC, Icecuber) are directly citable prior work.

- **Timeline:** Started Mar 25, 2026 → Entry/Team-Merger deadline **Oct 26, 2026** → Final Submission deadline **Nov 2, 2026** → Winners announced Dec 4, 2026.
- **Format:** Code Competition. Submissions run as a Kaggle Notebook, CPU or GPU ≤12h runtime, **no internet access**, external data/pretrained models allowed only if freely & publicly available. L4x4 (96GB) accelerators available at 2× GPU-quota cost vs. T4x2/P100.
- **Data:** Grids up to 30×30, ≤10 colors. Each task: 2–5 demonstration input/output pairs, then predict output(s) for unseen test input(s). Public training set: 1,000 tasks. Public evaluation set: 120 tasks. Private eval set size undisclosed.
- **Submission:** `submission.json`, exactly 2 attempts (`attempt_1`, `attempt_2`) per test output; scored as fraction where either attempt exactly matches ground truth.
- **Prizes ($700K total):** Progress Prizes $275K (top-8 leaderboard); **Grand Prize $275K** awarded to the best Solution Writeup, scored 0–5 on six equally-weighted criteria — Accuracy, Universality, Progress, Theory, Completeness, Novelty — writeup due within 7 days of the final deadline; Bonus Prize $150K split among top 5 teams if anyone hits 85% (won't be us, and doesn't need to be).
- **Open-source requirement:** prize-eligible entries must open-source system/model/weights under CC BY 4.0. Training from scratch sidesteps any pretrained-model licensing questions entirely.

**ARC Prize 2026 Paper Track** (arcprize.org/competitions/2026/paper, also listed as its own Kaggle competition `arc-prize-2026-paper-track`) — a separate, lower-barrier track. A paper must link to a *working* Kaggle code submission (it does **not** need a high score to qualify). Judged on the same 6-criteria rubric. Prizes: $75K guaranteed Top Paper (across ARC-AGI-2 and ARC-AGI-3 tracks) + $375K Outstanding Papers pool for anything scoring >4.5/5. **This is the primary target** — it rewards exactly the kind of controlled ablation study this project produces, independent of raw leaderboard rank. **Deadline confirmed via Kaggle API: Nov 9, 2026, 23:59 UTC.**

## 4. Technique inventory (from literature survey — what to build on / avoid re-treading)

| Approach | Mechanism | Verdict for this project |
|---|---|---|
| Icecuber (2020 winner) | Hand-crafted 142-fn DSL, greedy DAG search | Reference baseline only; ceiling ~20%, doesn't generalize past hand-enumerated primitives |
| Akyürek et al. TTT (arXiv:2411.07279) | Per-task fine-tuning at inference via self-generated augmentations | Don't re-implement — it's about *inference-time adaptation*, orthogonal to our *architecture* question |
| MindsAI (ARC Prize 2024 top score, 55.5%) | T5 backbone + synthetic pretrain + TTT | Not open-sourced, not reproducible, not our angle |
| **NSA: Neuro-symbolic ARC Challenge** (arXiv:2501.04424) | Transformer proposer (top-k candidate primitives) + DSL program search verified against ALL demo pairs; **correction from earlier read of the abstract only — the full paper does use test-time fine-tuning** (~2,500 synthetic per-task samples, 15 epochs, ~22 of a 30-min budget), it is not TTT-free | **Reuse the proposer+verifier pattern** (Phase 3), but our version deliberately omits TTT/RL to keep the ablation about *architecture*, not *inference-time adaptation* — an intentional scope difference from NSA, not an oversight |
| "ARC Is a Vision Problem!" / VARC (arXiv:2511.14761) | ViT over a rendered 64x64 canvas (2x2 patches) + per-task test-time training (~51 augmented tasks, 100 epochs, ~70s/task); ARC-1 60.4% (ensemble), ARC-2 11.1% | Confirms 2D/vision framing beats naive 1D tokenization for from-scratch ARC models — a stretch-goal representation ablation, not adopted as our default (we keep the flattened token sequence so all 3 backbones share one interface, per ROADMAP section 5.1) |
| "ARC-AGI Without Pretraining" / CompressARC (arXiv:2512.06104) | 76K-param equivariant network, no pretraining at all — minimizes description length (KL of a latent + reconstruction cross-entropy) purely at inference on the single target puzzle, 2000 Adam steps/puzzle (~20 min); ARC-1 eval 20% | Validates from-scratch training as legitimate and precedented; its equivariance-to-permutation/rotation/color trick is a candidate future ablation but out of scope for our controlled architecture comparison |
| Mission: Impossible Language Models (Kallini et al., ACL 2024, arXiv:2401.06416) | Trains GPT-2 from scratch on synthetic English vs. systematically-perturbed "impossible" variants (shuffle/reverse/hop classes); possible languages are learned faster (lower perplexity, higher surprisal-gap, earlier causal-intervention accuracy) — attributed to **information locality**, an inductive bias arising from the causal LM objective itself, not innate grammar knowledge | **Methodology template**: controlled synthetic-manipulation + multiple converging metrics (not just one accuracy number) to attribute a learning difference to a specific inductive bias rather than just reporting a score gap |

## 4.1 Hypothesis (for the paper's Abstract)

We hypothesize that a from-scratch backbone's positional inductive bias — not its raw parameter count or general sequence-mixing capacity — is the primary determinant of compositional generalization on ARC-AGI-2-style grid-reasoning tasks. Following the controlled synthetic-manipulation methodology of *Mission: Impossible Language Models* (Kallini et al., ACL 2024), which showed GPT-2 learns linguistically "possible" languages faster than matched "impossible" ones because of an information-locality bias rather than explicit grammatical knowledge, we built a systematic compositional split of ARC-style grid transformations and found an analogous architectural effect: a rotary-position-encoded (RoPE) Transformer generalizes to unseen (structural, color) primitive compositions markedly better than parameter-matched Mamba/SSM and Transformer+Mamba hybrid backbones trained under an identical regime (Phase 2b), and only the Transformer shows an interpretable train-eval-vs-held-out generalization gap at all — Mamba and the hybrid remain near floor regardless of composition novelty, suggesting they have not yet learned enough of the task to fail compositionally in an informative way. This architectural finding is orthogonal to, and composable with, two other recent from-scratch ARC directions: symbolic program verification (NSA's proposer+verifier, arXiv:2501.04424), which we show in Phase 3 can provably resolve any held-out composition that lies inside a small, enumerable transformation vocabulary, but does not by itself decide which neural backbone should propose candidates once the true transformation space is no longer enumerable; and representation-level changes such as treating grids as images (VARC, arXiv:2511.14761) or optimizing a per-task minimum-description-length objective at inference time (CompressARC, arXiv:2512.06104), both of which sidestep the sequence-backbone question by changing what the model computes *over* rather than *how it mixes sequence positions*. Holding representation, verifier, and training budget fixed, our contribution isolates a narrower question — which sequence-mixing operator's positional inductive bias is best suited to recombining previously-seen primitives into novel compositions — and our evidence so far points to explicit relative-positional structure (RoPE) over Mamba's implicit recurrent state as the deciding factor at this parameter scale.

## 5. Phased plan (Sep 6 – Nov 9, 2026, ~9 weeks)

### Phase 0 — Setup (Sep 6–12)
- [x] Accept competition rules on Kaggle (done for both `arc-prize-2026-arc-agi-2` and `arc-prize-2026-paper-track`).
- [x] Download ARC-AGI-2 public data (1,000 train + 120 eval tasks, plus a 240-task blind test set) into `data/`.
- [x] **Development environment decided: Kaggle Notebooks**, driven via the Kaggle API (`kaggle kernels push/pull/status/output`) from this repo, not local WSL2. Verified live: `mamba-ssm` + `causal-conv1d` build and import correctly on a Kaggle GPU T4 session using `pip install --no-build-isolation mamba-ssm causal-conv1d` (plain `pip install mamba-ssm` fails — its setup.py can't see the pre-installed `torch` inside pip's isolated build env). **Do not use the default P100 accelerator** — it's compute capability 6.0, below both the current PyTorch build's minimum (sm_70) and `mamba-ssm`'s own CUDA kernel requirement (sm_70+). Request T4x2 or the competition's L4x4 instead.
- [x] Confirmed data mount path inside a Kaggle session: `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/` (nested under `competitions/`, not flat).
- [ ] Stand up Michael Hodel's `arc-dsl` / an ARC-GEN-style generator to synthesize unlimited controlled tasks with known primitive composition.
- [ ] Read in full: Mission: Impossible Language Models, NSA, "ARC Is a Vision Problem!", "ARC-AGI Without Pretraining".
- [ ] Write a one-paragraph hypothesis statement (goes into the paper's Abstract later).

**Kaggle-kernel workflow reference** (established in `notebook/`):
- `notebook/kernel-metadata.json` + `notebook/main.py` (or additional scripts) are edited locally, then `kaggle kernels push -p notebook` uploads and runs a new version.
- Set `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json` to force T4x2 (the API also accepts `"NvidiaTeslaP100"` and `"Tpu1VmV38"`; L4x4 wasn't confirmed available through this field — use the web editor's Session Options if L4x4 is needed).
- Poll with `kaggle kernels status <owner>/<slug>` until `COMPLETE`.
- Pull logs with `kaggle kernels output <owner>/<slug> -p <dir>` — **on Windows this requires `PYTHONUTF8=1` set in the environment**, otherwise it crashes with a `charmap codec` error on any non-ASCII character in the log (e.g. pip's progress-bar characters). The downloaded `.log` file is a JSON array of `{"stream_name", "time", "data"}` records, not plain text — parse it to reconstruct readable output.
- The kernel's actual slug is whatever Kaggle resolves from the **title**, which can silently differ from the `id` you wrote in `kernel-metadata.json` if they don't match — check the URL in the push confirmation message and keep `id` in sync, or pushes will 409-conflict.

### Phase 1 — Controlled task generation (Sep 13–19)
- [x] Build synthetic task generator with **systematic compositional splits** (`generator.py`, vendoring `arc-dsl`): 4 structural x 4 color primitives, 12 train combos / 4 held-out diagonal combos, 600/200 tasks. Verified correct and leak-free.
- [x] Grid representation decided: flattened token sequence, row-major, `ROW_SEP` between rows, no explicit dimension tokens (recoverable by counting). 16-token vocab (10 colors + 6 structural tokens: `ROW_SEP, GRID_IN, GRID_OUT, PAIR_END, BOS, EOS`). Shared across all three backbones — see `arc_common/tokenizer.py`. 2D patch-embedding (vision-framing) stays a stretch-goal ablation, not the default representation.
- [x] Eval harness built (`arc_common/scoring.py`) implementing the exact ARC Prize metric (2 attempts/output, flat average across all test outputs, not per-task-then-averaged) plus `submission.json` template/validation. Self-tested in `test_harness.py`: tokenizer round-trips exactly on 3,600 real grids, perfect predictor scores 1.0, trivial baseline scores 0.0. This is the same code path that will score the real ARC-AGI-2 eval set and the final competition submission.

### Phase 2 — Baseline architectures (Sep 20 – Oct 3)
- [x] Implemented three **matched-parameter-count** (~1.0-1.08M params), from-scratch models in `notebook/phase2/phase2_train.py`: decoder-only Transformer, Mamba/SSM, interleaved Transformer+Mamba hybrid. Shared token embedding + learned positional embedding + weight-tied LM head across all three, so the only architectural difference under test is the sequence-mixing layer.
- [x] Trained on the 2,400 leave-one-out-augmented examples from the synthetic train split (80 epochs, AdamW, weight_decay=1e-2, dropout=0.1), with periodic checkpoint selection against a held-in generalization subsample (never the actual held-out combos) to guard against overfitting.
- [x] Evaluated via actual autoregressive generation (not just teacher-forced loss) + exact-match/cell-accuracy/shape-match scoring, on both `train_eval` (in-distribution) and `heldout` (compositional) synthetic splits.

**Pilot result (see `results/phase2_pilot_results.json`, reproduced across 2 independent full runs after fixing a checkpoint tie-break bug):**

| Model | Params | train_eval cell-acc | train_eval shape-match | heldout cell-acc | heldout shape-match |
|---|---|---|---|---|---|
| Transformer | 1.06M | 0.000 | 0.000 | 0.000 | 0.000 |
| Mamba | 1.08M | 0.046 | 0.075 | 0.053 | 0.085 |
| Hybrid | 1.01M | 0.045 | 0.075 | 0.045 | 0.070 |

Exact-match was 0.000 for all three (expected — exact whole-grid match is an extremely strict metric for this training budget; cell-accuracy and shape-match are the informative signals at this scale).

**Two findings, reported honestly and separately:**
1. **Robust:** under matched parameters and identical training budget, the vanilla decoder-only Transformer produced *zero* structurally valid grids at generation time (best checkpoint, teacher-forced loss 1.03), while Mamba (loss 0.47) and the hybrid (loss 0.57) both learned to produce partially-correct, correctly-shaped output a meaningful fraction of the time. This replicated identically across two independent full training runs.
2. **Null/preliminary, and important not to overclaim:** Mamba/hybrid's held-out (novel combination) scores are statistically indistinguishable from their in-distribution scores — i.e. at this pilot scale, we have NOT yet detected a *compositional-generalization-specific* gap. The finding so far is "does the architecture learn structured grid generation at all," not yet "does it specifically fail at recombining familiar primitives." Resolving whether a real compositional gap exists needs a properly resourced sweep (Phase 4) with more training and a working baseline for all three architectures first — a null result against an already-broken baseline (the Transformer) isn't informative.

**Debug follow-up (`notebook/phase2_debug/transformer_debug.py`, ~35 min run):** tested whether the Transformer's total failure was a fixable optimization issue. Trained the Transformer alone (skipping Mamba/hybrid to save time) under 3 configs: baseline (no warmup, lr=3e-4), +warmup+grad-clip at the same lr, and +warmup+grad-clip at lr=1e-3. Teacher-forced loss improved substantially with the higher LR (0.90 vs. baseline's 1.16 by epoch 39 — a real, meaningful optimization improvement) — **but quickcheck_cell_acc stayed at exactly 0.000 across all three configs, at every checkpoint.** This rules out learning-rate/warmup/gradient-clipping as the explanation: the model demonstrably fits the training objective better under tuning, yet still cannot produce a single structurally valid grid during free-running (non-teacher-forced) generation.

**Conclusion so far:** the failure is specific to autoregressive generation, not to training/optimization. This pointed toward either (a) an exposure-bias effect — self-attention conditioning on its own (early, likely-wrong) generated tokens compounds errors catastrophically in a way Mamba's recurrent state is more robust to — or (b) a genuine inductive-bias limitation of absolute positional embeddings + self-attention for this row-counting/copy-heavy structured task at small scale, where relative positional encoding might do much better.

**Debug follow-up 2 (`notebook/phase2_debug2/transformer_debug2.py`, ~49 min run):** tested both hypotheses together in a 2x2 design — {learned absolute pos embed, RoPE} x {plain teacher forcing, scheduled sampling ramped to p=0.5} — reusing the best regime from debug run 1 (lr=1e-3, warmup+clip, 40 epochs).

| Config | Params | best quickcheck_cell_acc |
|---|---|---|
| abs-pos, teacher-forcing | 1.06M | 0.000 |
| **RoPE, teacher-forcing** | 0.80M | **0.204** |
| abs-pos, scheduled-sampling | 1.06M | 0.000 |
| RoPE, scheduled-sampling | 0.80M | 0.085 |

**Resolved: hypothesis (b), not (a).** Swapping in RoPE alone — nothing else changed — took the Transformer from a flat 0.000 to 0.204 quickcheck cell-accuracy, still climbing at the final checkpoint. Scheduled sampling did not fix the abs-pos config (stayed at exactly 0.000) and actively *hurt* the RoPE config (0.204 -> 0.085). This rules out exposure bias as the (or even a helpful) fix, and confirms the failure was specifically about absolute positional embeddings being a poor fit for self-attention on this row-counting/copy-heavy structured-grid task at small scale. Two honest caveats: the RoPE config has ~25% fewer parameters (no learned position table to store) and uses a different attention implementation (manual QKV + `F.scaled_dot_product_attention` vs. `nn.MultiheadAttention`) than the abs-pos config, so this is a diagnostic result, not a perfectly isolated single-variable ablation — but the effect size (0.000 to 0.204) is large enough that it's not plausibly explained by either confound alone.

**Consequence for the project:** the Phase 2 pilot's Transformer row (0.000 across the board) is a fixable implementation artifact, not evidence about Transformers' inherent suitability for this task. The Phase 2 pilot needs a re-run with the Transformer using RoPE before the 3-architecture comparison is meaningful — this is now queued ahead of Phase 3.

### Phase 2b — corrected 3-architecture re-run (`notebook/phase2b/phase2b_train.py`, ~49 min run)

Re-ran the comparison with the confirmed fix: Transformer's self-attention uses RoPE (no additive positional embedding); Mamba unchanged; hybrid's Transformer sub-blocks also use RoPE while its Mamba sub-blocks keep the additive embedding (an explicitly flagged, not-yet-tested mix of both positional mechanisms). All three architectures also moved to the validated training regime (lr=1e-3, warmup, grad-clip) for a controlled comparison, not just the Transformer.

| Model | Params | train_eval cell-acc | train_eval shape-match | heldout cell-acc | heldout shape-match |
|---|---|---|---|---|---|
| **Transformer (RoPE)** | 0.80M | **0.309** | **0.492** | **0.235** | **0.380** |
| Mamba | 0.85M | 0.045 | 0.083 | 0.034 | 0.070 |
| Hybrid | 0.89M | 0.050 | 0.075 | 0.069 | 0.105 |

**This reverses the original Phase 2 picture.** With the only architecture-specific change being the Transformer's positional scheme (everything else, including the now-shared training regime, held constant across architectures), the RoPE-Transformer outperforms Mamba and the hybrid by roughly 6-7x on both splits — Mamba/hybrid's own scores barely moved from the original pilot. This is also the first time any architecture in this project has learned the task well enough to show an *interpretable* train_eval-vs-heldout gap rather than noise at floor: the Transformer's heldout scores drop ~23-24% relative to train_eval (cell-acc 0.309 -> 0.235, shape-match 0.492 -> 0.380), while Mamba/hybrid show no interpretable gap either way (their heldout scores are, if anything, comparable or nominally higher, but at absolute scores too low to be meaningful).

**Not yet a paper-ready claim (at time of writing above)** — three honest caveats: (1) this is a single run, and the original "Mamba/hybrid beat Transformer" claim was only trusted after 2 independent replications, so this reversal needs the same before being reported as robust; (2) the shared training regime was tuned via Transformer-focused debugging, not independently for Mamba/hybrid, so their weak scores may partly reflect a suboptimal regime for SSMs rather than a real ceiling; (3) the Transformer's own quickcheck score peaked at epoch 60 and fell by epoch 79 (checkpoint selection correctly caught this), suggesting it still degrades with more training in this regime at this scale — worth watching in Phase 4's longer runs.

**Replication (`notebook/phase2b_rep2/phase2b_rep2_train.py`, seed 1, ~50 min run): confirmed.**

| Model | Seed 0 train_eval / heldout cell-acc | Seed 1 train_eval / heldout cell-acc |
|---|---|---|
| **Transformer (RoPE)** | 0.309 / 0.235 | 0.275 / 0.191 |
| Mamba | 0.045 / 0.034 | 0.075 / 0.032 |
| Hybrid | 0.050 / 0.069 | 0.046 / 0.045 |

Both headline findings replicate under a genuinely different seed (different init + data-shuffling order, not just a re-run): the Transformer (RoPE) beats Mamba/hybrid by roughly 4-6x in both runs, and its train_eval-vs-heldout gap holds at a consistent magnitude (seed 0: cell-acc -23.9%, shape-match -22.8%; seed 1: cell-acc -30.5%, shape-match -29.0%). This now meets the same 2-independent-run bar the project held the original (reversed) pilot result to — **caveat (1) above is resolved; this is now a robust, reportable finding.** Mamba/hybrid still show no consistent train_eval-vs-heldout direction across seeds (hybrid's heldout was higher than train_eval in seed 0, roughly equal in seed 1) — consistent with "not learned well enough at this scale/regime for the comparison to be informative," independent of seed.

**Caveat 2 tested (`notebook/phase2c_mamba_tune/mamba_tune.py`, ~44 min run): resolved in favor of the current regime.** Trained Mamba alone under 3 regimes, matched exactly to Phase 2b's Mamba config: the current shared regime (lr=1e-3, warmup=0.05, clip=1.0, wd=1e-2), Mamba's original-pilot regime (lr=3e-4, no warmup/clip), and a no-weight-decay variant. Result: **the current regime won** (best quickcheck_cell_acc 0.034 vs. 0.019 for both alternatives) — reverting to Mamba's pre-fix regime or dropping weight decay made it *worse*, not better. Mamba's weak scores are not an artifact of a Transformer-biased training regime; the shared regime is the right choice for this comparison, and Phase 2b's numbers stand without a re-run.

**Where this leaves the central finding:** across 2 independent seeds and a 3-way hyperparameter sweep that ruled out the "unfair regime" explanation, the RoPE-fixed Transformer's ~4-6x advantage over Mamba/hybrid on this synthetic compositional-split task is solid enough to state directionally in the paper. Remaining caveat: this is one task family at one small parameter scale, and hybrid's own regime was never directly swept (lower priority now that Mamba's sweep suggests the shared regime isn't the limiting factor generally).

Full pipeline (data generation -> tokenization -> leave-one-out training -> checkpointed generation-based eval -> scoring) is now validated end-to-end on Kaggle. Two real bugs were caught and fixed along the way: a missing positional embedding (crippled the Transformer specifically, since self-attention needs it and SSM recurrence doesn't) and a checkpoint-selection tie-break bug (`>` instead of `>=`, which silently kept an untrained epoch-0 checkpoint when all evaluated scores tied at 0.0).

### Phase 3 — Neuro-symbolic verifier (Oct 4–10)
- [x] Built `arc_common/symbolic.py`: an NSA-style bounded program search (14 structural ops x ~136 color ops from `vendor/arc-dsl`) verified against every demonstration pair, keeping only provably-consistent candidates. Deliberately omits NSA's own test-time fine-tuning (see section 4 table correction) to keep the ablation about *architecture*, not *inference-time adaptation*.
- [x] Built `arc_common/tta.py`: cheap D4 (8-view rotation/reflection) test-time augmentation with majority-vote candidate combination, plus a `combined_predict` that tries the symbolic verifier first and falls back to TTA-ensembled neural generation.
- [x] Ran on Kaggle (`notebook/phase3/phase3_train.py`, 3,355s / ~56 min): retrained the same 3 architectures as Phase 2b and evaluated 4 conditions (baseline, tta_only on a 50-task subsample, symbolic_only, full_pipeline) on both synthetic splits.

**Results** (full numbers: `notebook/phase3/output/phase3_results.json`, log: `notebook/phase3/output/parsed.txt`):

| Model | Condition | train_eval cell-acc | heldout cell-acc |
|---|---|---|---|
| **symbolic_only** (shared, no neural model) | — | **1.000** | **1.000** |
| Transformer (RoPE) | baseline | 0.310 | 0.199 |
| Transformer (RoPE) | tta_only | 0.312 | **0.126** |
| Transformer (RoPE) | full_pipeline | 1.000 | 1.000 |
| Mamba | baseline | 0.075 | 0.047 |
| Mamba | tta_only | 0.093 | **0.000** |
| Mamba | full_pipeline | 1.000 | 1.000 |
| Hybrid | baseline | 0.055 | 0.062 |
| Hybrid | tta_only | 0.041 | 0.063 |
| Hybrid | full_pipeline | 1.000 | 1.000 |

**Finding 1 (expected, matches the pre-registered caveat in `arc_common/symbolic.py`): the verifier hits exactly its predicted ceiling.** `symbolic_only` and `full_pipeline` both score a perfect 1.000 for all three architectures, on both splits — the bounded search space is a strict superset of the generator's own primitives, so it provably finds a consistent program almost every time, making the underlying neural backbone's quality irrelevant once the verifier is attached. This is a **pipeline-correctness validation, not an architecture finding** — it confirms the full search-and-fallback mechanism works end-to-end on Kaggle, but does not generalize to the claim "the neuro-symbolic pipeline solves ARC": real ARC-AGI-2 (Phase 4) has a transformation space this small bounded search cannot exhaustively cover.

**Finding 2 (new, negative but informative): naive D4 test-time augmentation did not help, and hurt held-out generalization for 2 of 3 architectures.** Transformer's heldout cell-acc dropped 0.199 → 0.126 (-37% relative) under TTA; Mamba's collapsed to exactly 0.000; only the hybrid stayed flat. **Likely cause:** none of the three backbones were ever trained on rotated/mirrored views of a grid (the generator applies each task's structural primitive in one fixed orientation), so they have no rotation/reflection-equivariance to exploit — D4 TTA's core assumption (predictions should agree after undoing the symmetry) doesn't hold for a model that never saw those views, so majority-voting across 8 differently-wrong predictions can converge on a consistently-wrong answer rather than the right one. This suggests D4 TTA (used by several real prior ARC Kaggle solutions) is not a free lunch without matching training-time augmentation, as VARC (arXiv:2511.14761) explicitly uses — a concrete, testable follow-up rather than a reason to drop TTA from the paper.

**Finding 3 (replication): Transformer (RoPE) baseline again beats Mamba/hybrid ~4-6x** (0.310/0.199 vs. 0.075/0.047 and 0.055/0.062), consistent with Phase 2b and its seed-1 replication — a third independent confirmation of the central architectural finding. Minor caveat: this run's exact numbers differ slightly from Phase 2b's despite identical code and `torch.manual_seed(0)` (heldout 0.199 vs. 0.235) — CUDA kernel non-determinism (mamba-ssm / `scaled_dot_product_attention` backend selection) means same-seed reruns on Kaggle's shared GPUs aren't bit-reproducible, though the qualitative finding is unaffected.

### Phase 4 — Ablation & analysis (Oct 11–17) — *this is the paper's core contribution*
- Full grid: {3 architectures} × {with/without verifier} × {compositional-split severity}.
- Metrics: exact-match accuracy on held-out synthetic compositions, accuracy on the real ARC-AGI-2 public eval set (120 tasks), and qualitative failure analysis by primitive-combination type.
- Target output: a claim of the form "architecture X generalizes better to unseen compositions because Y" — not just a score table.

**Phase 4a — real ARC-AGI-2 accuracy, plus a KV-cache built first.** Real ARC-AGI-2 examples run far longer than the synthetic benchmark (median real-eval example ~2,700 tokens vs. ~200-800 synthetic), and the un-cached `generate()` used through Phase 3 is O(current_len^2) per generated token — the exact mechanism behind Phase 3's TTA runtime blowup, and worse here given the longer sequences. Rather than just cap sequence length lower and accept poor real-eval coverage, built a real KV-cache: `arc_common/models.py` extracts the shared Backbone/block code (previously duplicated across phase2b/phase3 scripts) and adds `forward_prefill`/`forward_step` — a true incremental cache for the RoPE attention path (O(current_len) per step instead of O(current_len^2)); Mamba's `forward_step` deliberately just re-runs its own already-linear-cost op over the cached pre-block sequence rather than wiring up `mamba_ssm`'s InferenceParams API, since there was no quadratic problem to fix there. **Verified mathematically exact on CPU before ever touching a GPU** (`test_phase4_kvcache.py`: cached generation's tokens *and* raw per-step logits match a non-cached reference exactly, across 5 random trials per architecture kind, using a fake-but-genuinely-causal substitute op for Mamba since the real kernel needs CUDA).

Measured real task-size distributions locally first: at a 4,096-token cap, 94.3% of training tasks and 80.0% of evaluation tasks fit, so results below are on that filtered 96/120-task eval subset, not the full public set — stated up front. Caught a real bug this way too, before Kaggle: leave-one-out training examples can be *longer* than a task's own eval prompt (holding out a train pair pulls the test pair into context), so filtering has to happen per-example, not per-task — found via a local dry run, fixed, and re-verified before any GPU time was spent.

Avoided the still-unresolved `storage.googleapis.com` Dataset-upload block (see Phase 3) a second way: rather than upload the real ARC-AGI-2 JSON files to our own Dataset, `notebook/phase4/kernel-metadata.json` adds `"competition_sources": ["arc-prize-2026-arc-agi-2"]` — Kaggle's own competition already hosts this exact data, so no upload was needed. A smoke-test push (2 epochs, tiny subsample) completed cleanly in 304s before committing to the full run, which completed in 4,139s (~69 min) — far faster than a rough pre-run estimate, since the KV-cache made eval much cheaper than expected.

**Results** (8 epochs, all 4,026 filtered leave-one-out training examples, full 96-task filtered eval set; full numbers: `notebook/phase4/output/phase4_results.json`):

| Model | Params | baseline exact | baseline cell-acc | baseline shape-match |
|---|---|---|---|---|
| symbolic_only (shared) | — | 0/139 | — | — |
| Transformer (RoPE) | 795,520 | 0.000 | **0.000** | **0.000** |
| Mamba | 1,022,336 | 0.000 | 0.027 | 0.036 |
| Hybrid | 1,103,872 | 0.000 | 0.061 | 0.101 |

(`with_verifier` == `baseline` for every architecture: the narrow curated struct+color DSL, as expected, essentially never matches real ARC-AGI-2's far more diverse transformations — 0/139 from the verifier alone, vs. a perfect 1.000 on the synthetic benchmark's enumerable space in Phase 3. Predicted, not a bug.)

**Reads like a reversal of the Phase 2b/3 finding, but the shape-match column shows what's actually happening: the Transformer isn't scoring low, it's producing zero structurally-valid output at all** — precisely the pre-RoPE collapse signature from the original Phase 2 pilot. Mamba (3.6% shape-match) and hybrid (10.1%) already produce *some* validly-shaped output at this same 8-epoch budget. **Most likely explanation: an under-training artifact, not a genuine architecture reversal or evidence against RoPE.** Phase 2b's own Transformer training curve took until epoch 60 of 80 to peak even on the much narrower, easier synthetic task family; this run used only 8 epochs on a harder, far more diverse real distribution — an order of magnitude less runway on a harder problem. **Resolved by a follow-up run: under-training, confirmed.** Reran Transformer-only (everything else identical) at 40 epochs instead of 8 (~110 min). The quickcheck trend makes the story visible directly: epochs 0 and 5 both still 0.000 (matching the 8-epoch run, which falls in that same dead zone), then epoch 10 jumps to 0.023 — first sign of life — climbing noisily to a best of 0.053 at epoch 35. Final full-eval numbers on that checkpoint: **cell-acc 0.079, shape-match 0.108** — up from a flat 0.000/0.000 at 8 epochs, and now slightly ahead of hybrid's own 8-epoch numbers (cell-acc 0.061, shape-match 0.101) despite hybrid having a 5x smaller epoch budget in that comparison. This confirms under-training, not a genuine reversal: the Transformer just needed a longer runway (matching Phase 2b's own synthetic training curve, which took until epoch 60 of 80 to peak) to leave the same zero-valid-output dead zone Phase 2's original pilot documented — it was never a "RoPE doesn't work on real data" story.

**Closed with a second follow-up: Mamba + hybrid reran at the same 40-epoch budget (~69 min).** Now a genuinely fair, matched-epoch 3-way real-ARC-AGI-2 comparison:

| Model | Params | cell-acc | shape-match |
|---|---|---|---|
| **Transformer (RoPE)** | 795,520 | **0.079** | **0.108** |
| Mamba | 1,022,336 | 0.047 | 0.072 |
| Hybrid | 1,103,872 | 0.044 | 0.065 |

**The Transformer's advantage transfers to real ARC-AGI-2 data, directionally replicating the synthetic-benchmark finding — but at a much smaller margin.** ~1.7x on cell-acc and ~1.5x on shape-match here, versus ~4-6x on the synthetic compositional benchmark. Two honest readings, not yet distinguished: (1) real ARC-AGI-2's far greater task diversity gives less room for RoPE's specific advantage (fixing a row-counting/copy-heavy structural bug) to dominate, since most real tasks aren't that narrow pattern; or (2) 40 epochs still isn't enough runway for the gap to fully open up — none of the three architectures' quickcheck curves had clearly plateaued (Mamba flatlined at 0.023 from epoch 5 to 35 then *dropped* to 0.000 at epoch 39, suggesting early convergence/overfitting; hybrid was still climbing at epoch 39, its best checkpoint; Transformer peaked at epoch 35 and dipped at 39) — a longer run could still change the picture, in either direction.

**Where this leaves Phase 4a:** the central architectural claim — RoPE-Transformer's inductive bias generalizes better than Mamba/hybrid's at this parameter scale — now has support on both the synthetic compositional benchmark (large margin, thoroughly replicated across seeds and hyperparameter sweeps) and real ARC-AGI-2 (smaller margin, one matched-epoch run). That's enough to state directionally in the paper with the appropriate caveat about margin size and training-budget sensitivity; a longer real-data run (all 3 architectures, more epochs) would strengthen the real-data claim further but is not required to report this honestly.

**Phase 4b — compositional-split-severity sweep on synthetic data.** The other half of Phase 4's original plan (`{3 architectures} x {compositional-split severity}`). Extended `generator.py` with `diagonal(k)`/`split_for_severity(severity)`: the 4x4 (structural x color) combo grid decomposes into 4 disjoint mod-4-shift "diagonals"; holding out `severity` of them (1, 2, or 3 out of 4) leaves fewer training combos and more held-out ones, while preserving the original design's key property at every level — every individual primitive still appears in at least one training combo (each row/column of the grid loses exactly `severity` of its 4 cells, never all 4), so held-out failure stays about novel *recombination*, not about primitives the model never saw. severity=1 exactly reproduces Phase 2b/3's original split (12 train / 4 heldout combos, verified byte-identical to the existing `data/synthetic/` files via a hash comparison after regenerating); severity=2 (8/8) and severity=3 (4/12) are new. Training task **count** is held fixed at 600 total across severities (`tasks_per_train_combo = 600 // n_train_combos`) so severity isolates compositional difficulty from training-data volume, not data quantity.

Ran `notebook/phase4b/phase4b_severity_train.py` on Kaggle (same Phase 2b model/training code, unchanged, reused verbatim since this experiment only varies the data): data for severity 2/3 is generated *on the fly* inside the kernel by calling `generate_split` from the already-mounted (unmodified) `generator.py` with an inlined copy of `diagonal`/`split_for_severity` — verified to match the canonical local version exactly before pushing, and this avoids needing another Dataset upload (the `storage.googleapis.com` block from Phase 3/4a is still unresolved). Completed in 9,178.3s (~153 min) for all 3 severities x 3 architectures x 80 epochs.

**Results** (full numbers: `notebook/phase4b/output/phase4b_results.json`):

| Severity | Combos (train/heldout) | Transformer train_eval cell / heldout cell | relative drop | Mamba heldout cell | Hybrid heldout cell |
|---|---|---|---|---|---|
| 1 (original) | 12 / 4 | 0.298 / 0.241 | 19.1% | 0.058 | 0.042 |
| 2 | 8 / 8 | 0.377 / 0.156 | 58.6% | 0.047 | 0.050 |
| 3 | 4 / 12 | **0.852** / **0.108** | **87.3%** | 0.040 | 0.037 |

(shape-match tells the same story: Transformer 0.458→0.370 [19.2% drop] at severity 1, 0.575→0.242 [57.9%] at severity 2, 0.925→0.222 [76.0%] at severity 3.)

**This is the clean, mechanistic result Phase 4 was designed to produce.** As compositional severity increases — fewer training combos, each seen with proportionally more data (600 total training tasks held fixed) — the Transformer's in-distribution accuracy climbs sharply (0.298 → 0.377 → 0.852 cell-acc: fitting 4 well-drilled combos is much easier than fitting 12 more sparsely-seen ones) while its held-out accuracy on genuinely novel (structural, color) pairings *falls* (0.241 → 0.156 → 0.108). The compositional generalization gap doesn't just exist, it **widens monotonically and dramatically with severity** — from a 19% relative drop to 87%. That's a direct, quantitative demonstration of *why* the architecture matters for compositional generalization specifically, not just a single score-table entry: the RoPE-Transformer is capable of fitting a narrow training distribution extremely well while specifically failing to recombine what it learned into novel combinations, and that failure gets worse precisely as fewer combinations are available to learn the compositional structure from.

Mamba and hybrid show the same *directional* trend (their heldout scores also decline as severity increases: Mamba 0.058→0.047→0.040, hybrid 0.042→0.050→0.037) but it's not clearly interpretable for either — their own train_eval scores at severity 3 (0.124, 0.125) are far below the Transformer's (0.852), meaning they never learn their own training combos well enough at this epoch budget for a "did it fail to generalize, or did it just fail to learn" distinction to be meaningful. This is consistent with every earlier finding in this project (Phase 2b, 2c, 4a): Mamba/hybrid are the "hasn't learned enough to be interpretable" architectures at this scale/budget, while the Transformer is the one whose successes and failures are both large enough to actually say something about.

### Phase 5 — Kaggle packaging (Oct 18–24)
- Package the best-performing pipeline as a self-contained Kaggle Notebook (≤12h runtime, no internet, only from-scratch weights — no external licensing questions).
- Do a dry-run submission early to catch Kaggle-specific runtime/environment issues before the deadline crunch.
- Open-source the repo (CC BY 4.0) — required for prize eligibility regardless of leaderboard placement.

**Started 2026-09-08.** `notebook/submission/submission.py`: Transformer (RoPE) only, no Mamba/hybrid — two reasons converge on this, not just one: (1) every experiment in this project (Phase 2b, 2c, 3, 4a, 4b) found it the strongest and most interpretable architecture, so submitting it keeps the paper and the submission consistent; (2) `mamba-ssm` needs `pip install` at kernel start, which needs internet, but a real scored run has none — dropping Mamba/hybrid sidesteps that constraint entirely rather than working around it. Kept the same ~800K-parameter scale used throughout the whole project rather than scaling up, since the "properly resourced" lever here is training *time* on real data (a wall-clock budget), not a bigger model — consistent with how Phase 4 already framed it, and keeps the submitted architecture identical to what the paper reports on.

Raised `FILTER_CAP` to 8,192 (from Phase 4a's 4,096) after measuring coverage locally first: 99.9% training / 98.3% public-eval / **100% blind-test** task coverage at this cap, versus 94.3%/80.0%/-- before — the blind test set is what's actually being predicted here, so its coverage matters more than it did for a research-only eval. Training is wall-clock-bounded (`MAX_TRAIN_SECONDS`, not a fixed epoch count) since per-epoch cost at this new, larger cap wasn't known in advance. Uses the KV-cached `generate()` (same `forward_prefill`/`forward_step` design as Phase 4a, inlined the same way) since blind-test prompts run up to 6,515 raw tokens — the un-cached generate() this project used through Phase 3 would be prohibitively slow here. The symbolic verifier (Phase 3, unchanged) runs first on every test example, falling back to the trained model only when no consistent program is found.

**Smoke test first** (120s training budget, 8-task blind-test subsample, `enable_internet: false` — a genuine no-internet dry run, not just a small one): completed cleanly in 185.4s. Confirmed: no internet needed anywhere in the pipeline, competition data (training/public-eval/blind-test, all three) mounts correctly, KV-cache runs without error, and a correctly-shaped `submission.json` is produced and passes `validate_submission`. All 8 smoke-test predictions fell back to a placeholder — expected, 120s of training produces no valid structured output yet, the same pattern seen everywhere else in this project at this little training.

**Full run** (6h training budget, full 240-task blind test set, `enable_internet: false`): completed in 22,355.4s (~6.2h total). Trained 90 epochs (538 steps/epoch on 4,298 leave-one-out examples), best checkpoint at epoch 76 (quickcheck_cell_acc 0.088). Public-eval sanity check: exact_match=0.000 over all 120 tasks (expected at this scale/budget — matches every real-ARC-AGI-2 result in Phase 4a, none of which broke past 0 exact-match either). **Final submission.json: valid, 240/240 tasks, 259/259 test outputs, 0 validation problems** (verified locally after download) — 2 outputs resolved by the symbolic verifier, 174 by the trained model, 83 fell back to a placeholder (structurally invalid model output, not a bug).

**One real (minor) bug found and fixed via the defensive fallback working as designed:** a single public-eval task's prompt alone exceeded `FILTER_CAP` (eval/blind-test prompts are never length-filtered — every test task needs *some* prediction, unlike training examples which can be safely dropped), pushing the total prefill length to 9,345 against a 9,344-slot RoPE cache and raising a shape-mismatch error. The existing `try/except` around generation caught it and fell back to a placeholder — no crash, no invalid submission, exactly the scenario that fallback was written for. Fixed properly rather than just leaving it tolerated: `MAX_SEQ_LEN`'s margin raised from 128 to 2,048 tokens, comfortably covering the actual observed max sequence length (9,306) across every real ARC-AGI-2 split. Not yet rerun with the fix — the existing submission.json remains valid and usable; the fix matters for Phase 6's actual final entry.

**Open-sourcing done 2026-09-08:** git-initialized the repo, added a `.gitignore` (excludes the real ARC-AGI-2 competition data for redistribution-licensing safety, build artifacts, local `.claude/` settings), a CC BY 4.0 `LICENSE`, and a `README.md`. Found and fixed one real issue before the first commit: `vendor/arc-dsl` had its own nested `.git` directory that git would have silently recorded as an empty embedded-repo reference instead of tracking its files — removed it so the vendored code tracks normally. Scanned all staged file contents for secrets before committing (clean). Pushed to https://github.com/Mayankz72/-arc-agi-architectural-bias (first push needed a merge — GitHub's own repo-creation flow had auto-generated a trivial placeholder README, resolved by keeping the real one). Phase 5 is now fully complete.

### Phase 6 — Final submission (Oct 25–31, buffer before Nov 2 deadline)
- Respect the 1-submission/day limit — don't attempt the final submission on the deadline day itself.
- Lock in the Final Submission selection (up to 2 allowed).

**First submission made 2026-09-08** (far ahead of the Oct 25–31 window — done now mainly to validate the end-to-end mechanism, not as the intended final entry). Confirmed via `kaggle competitions submit` that this is a Code Competition: the CLI upload was rejected outright (403, no submission consumed), meaning entry only happens through the notebook editor's "Submit to Competition" flow. Submitted "ARC-AGI 2 Submission - Version 2" (the run described in Phase 5, above) via Kaggle's web UI, driven with Claude in Chrome browser automation and explicit user confirmation before the final click — paused entirely, handing control back to the user, at a Kaggle identity-verification (Persona) step requiring their own webcam/smartphone action, since that's not something to do on someone's behalf. 1 of 2 final-leaderboard submission slots now used.

**Scored same day: 0.00 exact-match, "Succeeded."** Expected, not a bug — matches this project's own pre-submission public-eval sanity check (also 0.000, see Phase 5 above) and is consistent with ARC-AGI-2 being explicitly constructed to resist small, non-test-time-trained models (see Section 4.5 of `PAPER.md` for the full comparison against published approaches). The project's own graded metrics (cell-acc ~8%, shape-match ~11% from the same checkpoint) confirm real, if shallow, learning — exact whole-grid correctness is a much higher bar than a model this size with no test-time adaptation is likely to clear on a genuinely novel task.

**Remaining before the real Phase 6 window:** get the scored result back, optionally use the second slot on a rerun with the `MAX_SEQ_LEN` margin fix (Phase 5) or a better-trained model if the paper's later phases motivate one, then make the actual locked-in final selection near the Oct 25–31 window as originally planned.

### Phase 7 — Paper writeup (Nov 1–9)
- Write the paper in the organizer's required structure: Abstract, Intro, Prior Work, Approach, Results, Conclusion. Keep it short and equation-light per their explicit guidance ("shorter and clearer is always better").
- Cite prior ARC Prize solutions (Icecuber, MindsAI, ARChitects, NVARC, NSA, TTT) in Prior Work.
- Link the paper to the working Kaggle submission and submit to the Paper Track (`arc-prize-2026-paper-track` on Kaggle, deadline Nov 9, 2026 23:59 UTC).
- Polish the GitHub README for resume/portfolio use — this stands on its own even independent of the prize outcome.

**Draft written 2026-09-08** (far ahead of the Nov 1–9 window, same rationale as the early Phase 6 submission — validate the full pipeline works end-to-end this early, refine later). `PAPER.md` covers all six required sections plus a Reproducibility section, built entirely from findings already verified in this log — no new claims introduced while writing it. Cites Icecuber, Akyürek et al. TTT, MindsAI, NSA, ARChitects/NVARC, VARC, CompressARC, and Kallini et al. (Mission: Impossible LMs) in Prior Work, per the checklist above. Caught and fixed one real issue while writing: the abstract initially attributed specific, never-actually-verified author names to the NSA paper — corrected to cite by arXiv ID only (only Kallini et al.'s authorship was ever confirmed, via WebSearch, in an earlier session). README.md was written as part of open-sourcing the repo (Phase 5) rather than here, but satisfies this checklist item too.

**Not yet done:** proofread `PAPER.md` against the actual Paper Track submission form's format/length requirements (re-verify live, closer to when it's actually submitted, since competition details can change) and submit it there. The repository link placeholders are filled in (https://github.com/Mayankz72/-arc-agi-architectural-bias).

## 6. Stretch goals (only if time remains)
- Extend the architecture sweep to a 4th backbone (e.g. a linear-attention model like RWKV) for a broader claim.
- Adapt the pipeline to the sibling ARC-AGI-3 (interactive environments) track.

## 7. Known risks
- ~~`mamba-ssm` doesn't build on native Windows~~ — resolved by developing directly in Kaggle Notebooks via the API; confirmed working on GPU T4 with `--no-build-isolation`.
- Kaggle notebook compute is more restrictive than local dev (no internet during the actual scored submission run, 12h cap) — budget Phase 5 time for porting/debugging, don't leave it to the last days.
- Iterating via `kaggle kernels push` burns GPU quota (30h/week free tier) per run — batch changes into fewer, more complete pushes rather than pushing after every tiny edit.
