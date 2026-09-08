# Progress Log

Living log of what's actually been done on the ARC-AGI-2 architectural inductive-bias project. See `ROADMAP.md` for the full plan and rationale. Update this after each work session — newest entry on top.

---

### 2026-09-06 — Project kickoff
- Chose the project: a controlled study of architectural inductive bias (Transformer vs. Mamba/SSM vs. hybrid) for compositional generalization on ARC-AGI-2-style grid tasks, targeting the ARC Prize 2026 Paper Track.
- Verified live on Kaggle/arcprize.org: competition timeline (final submission Nov 2, 2026), Kaggle Notebook compute constraints (≤12h, no internet), dataset structure (1,000 train / 120 public eval tasks), and the Paper Track's 6-criteria rubric.
- Surveyed prior ARC-AGI approaches (Icecuber, TTT/MindsAI, NSA neuro-symbolic, "ARC Is a Vision Problem!") to confirm the architecture-comparison angle is a genuine gap, not a re-tread.
- Wrote `ROADMAP.md` with the full 9-week phased plan.
- **Status: Phase 0 (Setup) not yet started.**

### 2026-09-06 — Kaggle API connected + competitions joined
- Confirmed existing `kaggle.json` credentials at `~/.kaggle/` already work (CLI: `kaggle competitions list -s arc-prize`).
- Confirmed live: Paper Track (`arc-prize-2026-paper-track`) deadline is **Nov 9, 2026, 23:59 UTC**.
- Joined both `arc-prize-2026-arc-agi-2` and `arc-prize-2026-paper-track` via browser (rules accepted on both).
- Noted: Paper Track writeups become **public on Kaggle under CC BY 4.0** once the competition closes, attributed to Kaggle display name — factor this into what gets written into the paper (no confidential info).
- Kaggle API download still needs auth to actually pull files — untested past the 403-before-joining check.

### 2026-09-06 — Dataset downloaded
- Downloaded and unzipped `arc-prize-2026-arc-agi-2` into `data/`: `arc-agi_training_challenges.json` (1,000 tasks) + `arc-agi_training_solutions.json`, `arc-agi_evaluation_challenges.json` (120 tasks) + `arc-agi_evaluation_solutions.json`, `arc-agi_test_challenges.json` (240 tasks, blind — this is what a real submission predicts against), `sample_submission.json`.
- Verified task format: each task is `{"train": [{"input": grid, "output": grid}, ...], "test": [{"input": grid}, ...]}`, grids are small integer matrices.

### 2026-09-06 — Dev environment validated: Kaggle Notebooks via API
- Decided to develop directly on Kaggle (not WSL2) since the final submission has to be a Kaggle Notebook anyway, and Kaggle's environment already has a working CUDA toolchain for `mamba-ssm`.
- Scaffolded `notebook/` (kernel-metadata.json + main.py) and drove it entirely from the CLI: `kaggle kernels push/status/output`.
- First run failed twice, both fixed:
  - Default accelerator was GPU P100 (compute capability 6.0) — too old for both the installed PyTorch build and `mamba-ssm`'s own kernels (need sm_70+). Fixed by setting `"machine_shape": "NvidiaTeslaT4"` in kernel-metadata.json (discovered via the upgraded kaggle CLI 2.2.4 / kagglesdk source — not documented in the CLI's own help text).
  - `pip install mamba-ssm` failed at "Getting requirements to build wheel" — fixed with `pip install --no-build-isolation mamba-ssm causal-conv1d`.
- Second run: **fully green**. `mamba-ssm` imports and runs on GPU T4. Also found the real data mount path inside a session: `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/` (nested, not flat as assumed).
- Side note: `kaggle kernels output` crashes on Windows with a charmap codec error unless `PYTHONUTF8=1` is set first; upgraded kaggle CLI from 2.1.0 to 2.2.4 along the way.
- Also upgraded local `kaggle` package from 2.1.0 to 2.2.4.

### 2026-09-06 — Synthetic compositional-split generator built and verified
- Vendored Michael Hodel's `arc-dsl` (MIT licensed, compatible with our CC BY 4.0 obligation) into `vendor/arc-dsl/` for its grid-primitive functions.
- Wrote `generator.py`: a systematic compositional-generalization task generator, structured like a SCAN/COGS-style split. 4 structural primitives (`rot90`, `hmirror`, `vmirror`, `upscale2`) x 4 color primitives (`swap12`, `swap34`, `swap56`, `swap78`) = 16 combos. The 4 "diagonal" combos (e.g. `rot90`+`swap12`) are held out entirely from training — every individual primitive still appears in 3 other training combos, so the model can learn each one, but never sees that specific pairing until eval. This is the core experimental instrument for the whole project.
- Verified: composition semantics correct (independently recomputed `switch(rot90(grid), 3, 4)` and matched byte-for-byte), zero overlap between train/held-out combo sets, `upscale2` correctly doubles grid dimensions.
- Generated `data/synthetic/{train,heldout}_{challenges,solutions,combo_info}.json`: 600 train tasks (12 combos x 50), 200 held-out tasks (4 combos x 50).
- This all ran locally (pure Python, no GPU/Kaggle needed) — only actual model training needs to go to Kaggle.

### 2026-09-06 — Shared tokenizer + eval harness built
- `arc_common/tokenizer.py`: flattened row-major token sequence, 16-token vocab (10 colors + `ROW_SEP, GRID_IN, GRID_OUT, PAIR_END, BOS, EOS`). `encode_prompt`/`encode_target`/`decode_prediction` give the exact interface Phase 2's models will train against. No torch dependency — plain Python, works identically locally and in a Kaggle kernel.
- `arc_common/scoring.py`: implements the real ARC Prize metric precisely (2 attempts per test output, correct if either matches, flat average over all test outputs — not averaged per task first), plus `submission.json` template/validation helpers.
- `test_harness.py` self-tests both: round-trip exact on 3,600 grids from the synthetic set, perfect-predictor scores 1.0, trivial copy-input baseline scores 0.0. All passing.

### 2026-09-06 — Kaggle Dataset packaging + Phase 2 pilot training (7 kernel iterations)
- Discovered Kaggle kernel pushes only upload the single `code_file`, not sibling files — packaged `arc_common`, `generator.py`, the vendored DSL, and the synthetic data as a private Kaggle Dataset (`kojijhjughio/arc-agi-inductive-bias-shared`) that future kernels mount as input instead. Dataset mount path needed a recursive `os.walk` to find reliably (`/kaggle/input/datasets/<owner>/<slug>/...`, not the flatter path assumed at first).
- Built `notebook/phase2/phase2_train.py`: 3 matched-parameter-count (~1.0-1.08M) backbones (Transformer/Mamba/hybrid) trained from scratch on the synthetic compositional split, evaluated via real autoregressive generation + exact-match/cell-accuracy/shape-match scoring.
- Iterated through 7 kernel pushes fixing real bugs in order: (1) dataset path assumption wrong, (2) nested one level deeper than expected, (3) all 3 models scored exactly 0.0 on 15 epochs — just undertrained (loss was still dropping), (4) after 80 epochs, transformer completely degenerated (predicted background color forever, never emitted a row-separator) while mamba/hybrid partially worked — root cause: **no positional embedding was implemented at all**, which cripples self-attention (permutation-equivariant without it) but barely affects Mamba's inherently-sequential recurrence, (5) after adding positional embeddings, mamba/hybrid's training loss dropped sharply but their generalization metrics got *worse* — classic overfitting, fixed with dropout + weight decay + checkpoint selection against a held-in generalization subsample, (6) checkpoint-selection had a tie-break bug (`>` kept an untrained epoch-0 model when all scores tied at 0.0) — fixed to `>=`, (7) confirmed clean on a second independent run.
- **Final pilot result** (`results/phase2_pilot_results.json`): Transformer produces zero valid structured output even at its best checkpoint; Mamba and the hybrid both reach ~4.5-5.3% cell accuracy / ~7-8.5% shape-match, reproduced identically across 2 independent runs. Held-out (novel combination) and in-distribution scores are statistically indistinguishable for the two working architectures — no compositional-generalization-specific gap detected yet at this pilot scale. Full write-up and honest caveats in `ROADMAP.md` Phase 2 section.
- Each Kaggle kernel run took ~65-75 minutes end to end (T4 GPU, mostly autoregressive generation during eval, which has no KV-caching in this naive implementation).

### 2026-09-06 — Transformer failure debugged: ruled out LR/warmup, isolated to generation-time
- Built `notebook/phase2_debug/transformer_debug.py`: trains only the Transformer (skips Mamba/hybrid) under 3 configs to test the standard "needs LR warmup + grad clipping" hypothesis, faster (~35 min) than a full 3-architecture run.
- Result: higher LR (1e-3) + warmup + grad clipping improved teacher-forced training loss substantially (0.90 vs. baseline's 1.16 by epoch 39 -- a real optimization improvement) but **quickcheck_cell_acc stayed exactly 0.000 across all 3 configs at every checkpoint**. This rules out LR/warmup tuning as the fix.
- Conclusion: the failure is specific to free-running autoregressive generation, not to training/optimization -- likely exposure bias (self-attention compounding errors from its own generated tokens, which Mamba's recurrent state may be more robust to) or an absolute-positional-embedding limitation (relative encoding untested). Neither fully isolated; each remaining test is its own ~40-70 min experiment. Documented as an honest open question in `ROADMAP.md` rather than pursued further immediately given time invested this session.
- Total Phase 2 GPU time this session: 7 phase2_train.py iterations (~65-75 min each) + 1 debug run (~35 min) ~= 8+ hours of Kaggle T4 quota used (well within the 30h/week free tier).

### 2026-09-06 — Debug follow-up 2: RoPE x scheduled-sampling 2x2, launched
- Built `notebook/phase2_debug2/transformer_debug2.py` to isolate the two hypotheses left open by the first debug run (exposure bias vs. absolute-positional-embedding limitation), rather than testing them one at a time.
- 2x2 design: {learned absolute pos embed, RoPE} x {plain teacher forcing, scheduled sampling ramped linearly to p=0.5}. All 4 configs otherwise reuse the best regime from debug run 1 (lr=1e-3, warmup_frac=0.05, grad_clip=1.0, weight_decay=1e-2, 40 epochs).
- RoPE required a manual QKV attention implementation (`CausalSelfAttentionRoPE`, using `F.scaled_dot_product_attention` with `is_causal=True`) since `nn.MultiheadAttention` (used for the abs-pos configs, unchanged from debug run 1) doesn't expose a hook to rotate Q/K before the dot product. Caveat to keep in mind when interpreting results: the abs-pos and RoPE configs therefore differ in attention implementation details beyond just the positional scheme (different weight init shape/scale for QKV projections), not a perfectly isolated ablation.
- Scheduled sampling implemented as the standard two-forward-pass approximation: a no-grad forward pass to get the model's own greedy predictions, then a second forward pass where target-region input tokens are replaced by those predictions with probability `p` (only this second pass's loss is backpropped). Cheaper than true autoregressive rollout during training.
- Pushed to Kaggle as `kojijhjughio/arc-agi-2-transformer-debug2`. Awaiting results — see next entry once complete.

### 2026-09-06 — Debug follow-up 2: results — RoPE alone fixes the Transformer
- Run completed in 2,943s (~49 min), matching the estimated 40-70 min. Full numbers: `notebook/phase2_debug2/output/transformer_debug2_results.json`, log: `notebook/phase2_debug2/output/parsed.txt`.

| Config | Params | best quickcheck_cell_acc |
|---|---|---|
| abs-pos, teacher-forcing (repeat of prior best) | 1.06M | 0.000 |
| **RoPE, teacher-forcing** | 0.80M | **0.204** |
| abs-pos, scheduled-sampling (ramp to p=0.5) | 1.06M | 0.000 |
| RoPE, scheduled-sampling (ramp to p=0.5) | 0.80M | 0.085 |

- **Clean result: RoPE alone resolves the failure.** Swapping the learned absolute positional embedding for RoPE — with everything else identical (same lr/warmup/clip/weight-decay regime as debug run 1's best config) — took quickcheck_cell_acc from a flat 0.000 to 0.204, and it was still climbing at the final checkpoint (0.041 at epoch 30 -> 0.204 at epoch 39), suggesting more epochs would likely help further.
- **Scheduled sampling did not help, and actively hurt where anything was working.** abs-pos stayed at exactly 0.000 regardless of scheduled sampling (ruling out exposure bias as *the* fix for the abs-pos failure specifically), and adding scheduled sampling to the RoPE config made it *worse* (0.204 -> 0.085), not better.
- **Conclusion: hypothesis (b) confirmed, (a) not supported.** The Transformer's total generation-time failure in the Phase 2 pilot was caused by the learned absolute positional embedding being a poor fit for this row-counting/copy-heavy structured-grid task, not by exposure bias. This is now a resolved, positive finding rather than an open question — safe to report in the paper's Theory section as: self-attention with absolute positional embeddings catastrophically fails at free-running structured grid generation at this scale, and this is specifically a positional-encoding effect (fixed by RoPE), not a general self-attention-vs-generation problem.
- **Caveat:** RoPE config has ~25% fewer parameters (0.80M vs 1.06M) than the abs-pos config purely because RoPE has no learned position-embedding table to store (2048 x 128 = 262,144 params) — the comparison here is diagnostic, not a matched-parameter-count claim like Phase 2's main comparison. Also, the RoPE attention module is a different implementation (manual QKV + `F.scaled_dot_product_attention`) than the abs-pos config's `nn.MultiheadAttention`, so the ablation isn't perfectly isolated to "just the positional scheme" — a second, minor confound worth flagging honestly rather than hiding.
- **Implication for the project:** the Phase 2 pilot's Transformer baseline should be re-run with RoPE before drawing any final architecture-comparison conclusions — the current Phase 2 table's Transformer row (0.000 across the board) reflects a fixable implementation choice, not an inherent Transformer limitation, and isn't a fair comparison against Mamba/hybrid as it stands.

### 2026-09-06 — Phase 2b: corrected 3-architecture re-run, launched
- Built `notebook/phase2b/phase2b_train.py`, a corrected re-run of the Phase 2 pilot applying the fix confirmed in the debug follow-ups: the Transformer's self-attention now uses RoPE (`CausalSelfAttentionRoPE`, copied from `phase2_debug2`) with no additive positional embedding at all. Mamba is unchanged (kept its additive learned positional embedding, since it already worked). Hybrid's Transformer sub-blocks also switched to RoPE (same fix — they do self-attention too), while its Mamba sub-blocks keep the additive embedding — an explicitly flagged caveat: hybrid now mixes both positional mechanisms, an untested combination, though architecturally sensible (each mixing operator gets the scheme it needs).
- Also upgraded the training regime for **all three** architectures (not just the transformer) to the validated debug regime — lr=1e-3, warmup_frac=0.05, grad_clip=1.0 — so the comparison stays controlled (identical regime across architectures) while using the better-performing regime now that it's confirmed not to hurt anything. Epoch budget (80) and everything else unchanged from the original Phase 2 pilot for a like-for-like re-run.
- Pushed to Kaggle as `kojijhjughio/arc-agi-2-inductive-bias-phase-2b-train`. Awaiting results — see next entry once complete.

### 2026-09-06 — Phase 2b results: RoPE-fixed Transformer now dominates Mamba/hybrid
- Run completed in 2,953s (~49 min). Full numbers: `notebook/phase2b/output/phase2b_results.json`, log: `notebook/phase2b/output/parsed.txt`.

| Model | Params | train_eval cell-acc | train_eval shape-match | heldout cell-acc | heldout shape-match |
|---|---|---|---|---|---|
| **Transformer (RoPE)** | 0.80M | **0.309** | **0.492** | **0.235** | **0.380** |
| Mamba | 0.85M | 0.045 | 0.083 | 0.034 | 0.070 |
| Hybrid | 0.89M | 0.050 | 0.075 | 0.069 | 0.105 |

(Exact-match is 0.000 for all three, as in the original pilot — still an extremely strict metric at this training budget.)

- **This is a complete reversal of the original Phase 2 pilot's picture.** Before the fix, Mamba/hybrid "worked" (weakly) and the Transformer was totally broken (0.000). Now, with the *only* architecture-specific change being the Transformer's positional scheme (RoPE instead of additive absolute embedding — Mamba's positional handling is untouched) and an *identical* upgraded training regime applied to all three (lr=1e-3, warmup, grad-clip), the RoPE-Transformer outperforms Mamba and the hybrid by roughly 6-7x on cell-accuracy and shape-match, on both splits. Mamba and hybrid's own scores are close to their original Phase 2 pilot numbers (0.045-0.05 cell-acc, same ballpark) — they weren't hurt or helped much by the training-regime upgrade; the Transformer was transformed by fixing its actual bug.
- **First real sign of a compositional-generalization-specific gap, though preliminary:** the Transformer's heldout scores (novel primitive combinations) are consistently lower than its train_eval scores — cell-acc 0.309 -> 0.235 (~24% relative drop), shape-match 0.492 -> 0.380 (~23% relative drop). This is the first time in the project any architecture has learned the task well enough to make a train_eval-vs-heldout gap *interpretable* rather than noise at floor. Mamba/hybrid still show no interpretable gap (differences are within noise of their low absolute scores, and heldout is even nominally higher than train_eval for hybrid, i.e. no signal either way).
- **Honest caveats, not yet a paper-ready claim:**
  1. This is a single run — the original Phase 2 pilot's "Mamba/hybrid beat Transformer" claim was only trusted after 2 independent replications; this reversal needs the same treatment before going in the paper.
  2. The training regime (lr, warmup, grad-clip) was tuned via Transformer-focused debugging, not independently tuned for Mamba/hybrid — their weak scores could partly reflect a regime that's suboptimal for SSMs specifically, not a ceiling on what Mamba can do here. A fair final comparison (Phase 4) should tune each architecture's own hyperparameters rather than reusing one shared debugged-for-Transformer regime.
  3. The Transformer's own quickcheck_cell_acc peaked at epoch 60 (0.191) and *fell* by epoch 79 (0.123) — checkpoint selection correctly picked epoch 60, but this is a sign the Transformer still degrades with more training at this scale/regime; worth watching in Phase 4's longer runs.
- **Bottom line for now:** architecture clearly does matter at this scale once each backbone's known failure modes are fixed — but which way the *causal* story runs (self-attention+RoPE is just better at this task; or Mamba/hybrid are specifically undertrained here) isn't settled yet. Needs replication + per-architecture hyperparameter tuning before the paper states a real directional claim.

### 2026-09-06 — Phase 2b replication run 2, launched
- Built `notebook/phase2b_rep2/phase2b_rep2_train.py`: identical to `phase2b_train.py` except `torch.manual_seed(0)` -> `torch.manual_seed(1)` everywhere (different random init + data-shuffling order), to test whether Phase 2b's reversal (RoPE-Transformer beating Mamba/hybrid ~6-7x) replicates under a genuinely different seed, not just a re-run of the same one.
- Pushed to Kaggle as `kojijhjughio/arc-agi-2-inductive-bias-phase-2b-rep2-train`. Awaiting results.

### 2026-09-06 — Phase 2b replication results: reversal and generalization gap both confirmed
- Run completed in 2,970s (~50 min). Full numbers: `notebook/phase2b_rep2/output/phase2b_rep2_results.json`, log: `notebook/phase2b_rep2/output/parsed.txt`.

| Model | Seed 0 (phase2b) train_eval / heldout cell-acc | Seed 1 (rep2) train_eval / heldout cell-acc |
|---|---|---|
| **Transformer (RoPE)** | 0.309 / 0.235 | 0.275 / 0.191 |
| Mamba | 0.045 / 0.034 | 0.075 / 0.032 |
| Hybrid | 0.050 / 0.069 | 0.046 / 0.045 |

(shape-match tells the same story: Transformer 0.492/0.380 seed 0 vs. 0.458/0.325 seed 1; Mamba/hybrid stayed in the 0.06-0.13 range both seeds.)

- **Both Phase 2b findings replicate under an independent seed.** The Transformer (RoPE) beats Mamba and hybrid by a wide margin in both runs — roughly 6x seed 0, roughly 4-6x seed 1 depending on which baseline — confirming this is a real architectural effect, not a one-seed fluke. This now meets the same 2-independent-run bar the project already held the original (reversed) Phase 2 pilot result to, and can be reported as a real finding rather than preliminary.
- **The Transformer's train_eval-vs-heldout gap also replicates, at a consistent magnitude.** Seed 0: cell-acc dropped 23.9% (0.309->0.235), shape-match 22.8% (0.492->0.380). Seed 1: cell-acc dropped 30.5% (0.275->0.191), shape-match 29.0% (0.458->0.325). Both seeds show a real, same-direction, similar-magnitude (~23-31%) degradation from in-distribution to novel compositions — this is now a credible signal of an actual compositional-generalization gap for the Transformer, the first such signal in the project.
- **Mamba/hybrid still show no consistent gap direction across seeds** — e.g. hybrid's heldout was *higher* than train_eval in seed 0 (0.069 vs 0.050) but roughly equal in seed 1 (0.045 vs 0.046); Mamba's train_eval score itself varied 0.045->0.075 between seeds while heldout stayed ~0.03 both times. This reinforces the earlier read: Mamba/hybrid haven't learned the task well enough at this scale/regime for a train_eval-vs-heldout comparison to be informative either way, independent of seed.
- **Remaining caveat unchanged:** the shared training regime was tuned via Transformer-focused debugging, not independently for Mamba/hybrid. The *reversal itself* is now confirmed robust to seed, but whether Mamba/hybrid could close the gap under their own tuned hyperparameters is still open — that's the next thing to check before drawing a final "architecture X is better because Y" claim for the paper.

### 2026-09-06 — Mamba hyperparameter tuning sweep, launched
- Built `notebook/phase2c_mamba_tune/mamba_tune.py` to directly test the remaining open caveat: is Phase 2b's shared training regime (tuned to fix the Transformer's bug) actually suboptimal for Mamba specifically? Trains ONLY Mamba (skips Transformer/hybrid to save time), matched exactly to Phase 2b's Mamba config (same 848,256-param architecture, same 80-epoch budget, same data), under 3 regimes:
  1. current (phase2b's regime): lr=1e-3, warmup=0.05, clip=1.0, wd=1e-2
  2. original-pilot regime (what Mamba used before the Transformer fix existed): lr=3e-4, no warmup, no clip, wd=1e-2
  3. no weight decay: same as (1) but wd=0.0 — tests whether weight decay specifically hurts Mamba's recurrent state-space parameters
- Pushed to Kaggle as `kojijhjughio/arc-agi-2-mamba-tune`. Awaiting results — if a config clearly beats the current regime, hybrid should get the same sweep before Phase 4; if not, the current shared regime stands and the reversal's cause is architectural, not a tuning artifact.

### 2026-09-06 — Mamba tuning results: current regime already wins, caveat resolved
- Run completed in 2,666s (~44 min). Full numbers: `notebook/phase2c_mamba_tune/output/mamba_tune_results.json`, log: `notebook/phase2c_mamba_tune/output/parsed.txt`.

| Config | best quickcheck_cell_acc |
|---|---|
| **current (phase2b regime): lr=1e-3, warmup=0.05, clip=1.0, wd=1e-2** | **0.034** |
| original-pilot regime: lr=3e-4, no warmup, no clip, wd=1e-2 | 0.019 |
| no weight decay: lr=1e-3, warmup=0.05, clip=1.0, wd=0.0 | 0.019 |

- **The regime Phase 2b already used for Mamba is the best of the three tested — not a handicap.** Neither reverting to Mamba's original lower-lr/no-warmup regime nor removing weight decay improved on the current shared regime; both scored roughly 45% lower. This directly answers the open caveat from Phase 2b: **Mamba's weak scores are not an artifact of a Transformer-biased training regime.** The current shared regime stands as the right choice for the 3-architecture comparison.
- Side observation, consistent with the Transformer's own late-training wobble: the current-regime config's quickcheck_cell_acc was non-monotonic across checkpoints (0.000 -> 0.025 -> 0.034 -> 0.000 -> 0.000), i.e. noisy/near-floor at this quick-subsample scale regardless of config — checkpoint selection (picking epoch 40's 0.034) correctly caught the peak, consistent with how Phase 2b/rep2's own Mamba checkpoints were chosen.
- **Consequence:** no re-run of Phase 2b is warranted — its numbers stand. Given the current regime already wins the sweep, tuning hybrid's regime separately is now lower-priority (deprioritized on the Next-up list, not dropped) — the shared regime not being the limiting factor for Mamba makes it less likely to be one for hybrid either, though hybrid was never directly tested.
- **Where this leaves the project's central finding:** across 2 independent seeds (Phase 2b + rep2) and now a 3-way hyperparameter sweep ruling out the "unfair regime" explanation, the RoPE-fixed Transformer's ~4-6x advantage over Mamba/hybrid on this synthetic compositional-split task is a real architectural effect at this scale, not a tuning artifact. That's now solid enough to state directionally in the paper (with the still-open caveat that this is one task family, one parameter scale, and hybrid's own regime remains untested).

### 2026-09-07 — Phase 3 verifier layer built + validated locally; read the 4 core papers; hypothesis paragraph written; blocked on a Kaggle upload

- Built `arc_common/symbolic.py`: an NSA-style (arXiv:2501.04424) bounded program search over a curated subset of `vendor/arc-dsl` (14 structural ops x ~136 color ops), verified against every demo pair in a task. Built `arc_common/tta.py`: D4 (8-view) test-time-augmentation majority voting, plus `combined_predict` (symbolic first, TTA-ensembled neural fallback). Both are pure Python, no torch dependency, so they're unit-testable locally.
- `test_phase3_harness.py` (new, mirrors `test_harness.py`'s style): validated locally, no GPU needed.
  - First run: `symbolic_predict` solved only 75% of `train_eval` (90/120) — root cause: the curated structural-op library omitted `upscale2`, one of the generator's 4 structural primitives (3/4 covered => exactly 75%, confirming the diagnosis). Fixed by adding `upscale2`/`upscale3`/`downscale2` to `STRUCT_OPS`.
  - After the fix: 100% exact-match on both `train_eval` (120/120) and `heldout` (200/200) — expected and documented as an honest caveat in the module docstring: the search space is a strict superset of the generator's own primitives, so this is a pipeline sanity check / upper bound, not evidence about real ARC-AGI-2 (Phase 4), where the transformation space isn't exhaustively enumerable by a search this small.
  - Also verified: symbolic search correctly abstains (`None`) when no consistent program exists; D4 forward/inverse transforms round-trip exactly; TTA majority vote recovers the right answer when all 8 views agree; `combined_predict` correctly overrides a deliberately-wrong neural stub with the symbolic verifier's (correct) answer, and correctly falls through to the wrong neural stub when `use_symbolic=False`.
- Built `notebook/phase3/phase3_train.py`: retrains the same 3 architectures as Phase 2b (identical code/hyperparameters/seed) and evaluates 4 conditions — baseline (greedy, no verifier), `tta_only` (8-view majority vote, bounded to a 50-task subsample per split since it's 8x the generation cost), `symbolic_only` (shared across architectures, full splits, no GPU needed), and `full_pipeline` (symbolic first, TTA-neural fallback, full splits — expected near-zero fallback rate given the symbolic search's ~100% synthetic coverage).
- Read all 4 core papers from the Next-Up list in full (via arXiv HTML, not just abstracts) and wrote the paper's hypothesis paragraph — see `ROADMAP.md` section 4.1. **Correction caught while doing this:** ROADMAP's technique-inventory table previously said NSA uses "no TTT/RL" — the full paper actually describes extensive test-time fine-tuning (~2,500 synthetic per-task samples, 15 epochs, ~22 of a 30-minute budget) that *dominates* its compute. Fixed the table entry and explicitly noted our Phase 3 verifier deliberately omits TTT/RL as a scope choice, not an oversight.
- **Blocked, root-caused:** pushing the updated shared Kaggle Dataset (needed so the Phase 3 kernel can `import arc_common.symbolic`/`arc_common.tta`) failed on 4 separate attempts (`kaggle datasets version`, both `--dir-mode zip` and `--dir-mode tar`, plus a detached retry) with an identical, fully-reproducible signature: transfer rate starts fast then decays smoothly toward a stall, never erroring out. Diagnosed with `curl -v https://storage.googleapis.com/` (this is where Kaggle's dataset uploads actually go, via a signed resumable-upload URL): **TCP connects and the TLS handshake completes normally, the HTTP GET is sent, then zero bytes ever come back — a 15s timeout with no response.** Meanwhile `kaggle.com`'s own API (`kaggle datasets list`, ~2s), `www.googleapis.com` (0.3s), plain ICMP ping to `storage.googleapis.com` (16ms, 0% loss), and a raw 300KB POST upload to `httpbin.org` (65KB/s, fully normal) all work fine. This is the signature of network-level filtering that allows a TLS handshake to `storage.googleapis.com` specifically but silently black-holes the response after — not a bandwidth problem, not a Kaggle credentials/API problem, and not fixable by retrying, changing `--dir-mode`, or any client-side flag. **Next session: try from a different network** (mobile hotspot, VPN, different machine) before attempting the Phase 3 Kaggle push again — if a browser-based upload through kaggle.com's web UI also silently fails/hangs on this same network, that confirms it's this network's firewall/filtering blocking `storage.googleapis.com`, not specific to the CLI.

### 2026-09-07 — Phase 3 results: symbolic verifier hits its predicted ceiling, TTA alone is a wash-to-negative

**Unblocked without needing the stalled dataset push**: rather than keep fighting the `storage.googleapis.com` network block, inlined `arc_common/symbolic.py` + `arc_common/tta.py`'s logic directly into `notebook/phase3/phase3_train.py` (verified byte-for-byte equivalent locally by exec'ing the extracted block against the real synthetic data: 120/120 and 200/200, matching the standalone modules). This only needed `kaggle kernels push`, which doesn't touch that endpoint and pushed in 2.5s. Lesson: `kaggle kernels push` and `kaggle datasets version` use different upload paths — a blocked one doesn't imply the other is blocked too.

Run completed in 3,355s (~56 min actual compute — the wall-clock gap before `kaggle kernels status` reported `COMPLETE` was much longer, ~3.5h, apparently a Kaggle-side queuing/reporting delay, not actual runtime; `kaggle kernels logs` was able to fetch the full completed log before `status` had updated, which is the more reliable way to check a long-running kernel). Full numbers: `notebook/phase3/output/phase3_results.json`, log: `notebook/phase3/output/parsed.txt`.

| Model | Condition | train_eval cell-acc | heldout cell-acc |
|---|---|---|---|
| **symbolic_only** (shared, no neural model) | -- | **1.000** | **1.000** |
| Transformer (RoPE) | baseline | 0.310 | 0.199 |
| Transformer (RoPE) | tta_only (8-view D4, 50-task subsample) | 0.312 | **0.126** |
| Transformer (RoPE) | full_pipeline (symbolic + TTA fallback) | 1.000 | 1.000 |
| Mamba | baseline | 0.075 | 0.047 |
| Mamba | tta_only | 0.093 | **0.000** |
| Mamba | full_pipeline | 1.000 | 1.000 |
| Hybrid | baseline | 0.055 | 0.062 |
| Hybrid | tta_only | 0.041 | 0.063 |
| Hybrid | full_pipeline | 1.000 | 1.000 |

**Finding 1 (expected, matches the pre-registered caveat): the symbolic verifier hits exactly its predicted ceiling.** `symbolic_only` and `full_pipeline` both score a perfect 1.000 on every metric, for all three architectures, on both splits. This confirms the honest caveat written into `arc_common/symbolic.py` *before* this run: since the bounded (struct x color) search space is a strict superset of the generator's own primitives, it provably finds a consistent program for essentially every task, making the neural backbone's own quality irrelevant once the verifier is attached. **This is a pipeline-correctness result, not an architecture result** — it validates that the full neuro-symbolic mechanism (search, consistency-check, fallback wiring) works end-to-end, but it cannot be reported as "the hybrid pipeline solves ARC" without the caveat that the real ARC-AGI-2 transformation space (Phase 4) is not exhaustively enumerable by a search this small.

**Finding 2 (new, genuinely informative negative result): naive 8-view D4 test-time augmentation did not help, and actively hurt held-out generalization for 2 of 3 architectures.** Transformer's heldout cell-acc dropped 0.199 -> 0.126 (-37% relative) under TTA; Mamba's collapsed 0.047 -> 0.000; only the hybrid was roughly flat (0.062 -> 0.063). train_eval scores were a mixed wash (Transformer and Mamba nudged up slightly, hybrid down slightly) -- nothing like the clean win TTA is expected to give. **Likely cause, not yet confirmed:** none of the three backbones were ever trained on rotated/mirrored/transposed views of a grid -- `generator.py` always applies its structural primitive in one fixed canonical orientation per task, so the models have no rotation/reflection-equivariance inductive bias to exploit. D4 TTA's core assumption -- that the "right answer" should look the same after undoing a symmetry transform -- doesn't hold for a model that was never shown those transformed views during training, so majority-voting across 8 mostly-wrong-in-different-ways predictions doesn't converge on the right answer; it can converge on a consistently-wrong one instead (explaining Mamba's heldout going to exactly 0.000 rather than just noisier). **Implication:** cheap D4 TTA (a technique several real prior ARC Kaggle solutions use) is not a free lunch here -- it likely only pays off for a model trained with matching rotation/reflection augmentation (as VARC, arXiv:2511.14761, explicitly does via its own augmentation pipeline), which none of our Phase 2/2b/3 models use. Worth flagging as a concrete, testable follow-up rather than silently dropping TTA from the paper.

**Finding 3 (replication, third independent run of the core result): Transformer (RoPE) baseline again clearly beats Mamba/hybrid** -- 0.310/0.199 vs. 0.075/0.047 (mamba) and 0.055/0.062 (hybrid), a ~4-6x margin consistent with Phase 2b and its seed-1 replication. Note the exact numbers aren't bit-identical to Phase 2b's seed-0 run despite identical code and `torch.manual_seed(0)` calls (this run: heldout 0.199 vs. Phase 2b's 0.235) -- CUDA kernel non-determinism (mamba-ssm and/or `scaled_dot_product_attention`'s backend selection) means same-seed reruns aren't bit-reproducible on Kaggle's shared GPUs, a minor caveat worth noting in the paper's reproducibility section, though it doesn't change the qualitative finding.

### 2026-09-07 — Phase 4a: KV-cache built + verified, then real ARC-AGI-2 training/eval run

**Scoping decision (user-confirmed via AskUserQuestion):** given real ARC-AGI-2 tasks run 2-8x longer than our synthetic benchmark (median real-eval example ~2,700 tokens vs. ~200-800 synthetic) and the naive `generate()` used through Phase 3 recomputes full self-attention over the whole growing sequence at every generated token (O(current_len^2) per step -- exactly what caused Phase 3's TTA blowup), chose to build a proper KV-cache before attempting real-data eval, rather than just capping sequence length lower and accepting much worse real-eval coverage.

- Built `arc_common/models.py`: extracted the shared Backbone/TransformerBlock/MambaBlock/CausalSelfAttentionRoPE code (previously copy-pasted across phase2b/phase3 scripts) into one module, and added `forward_prefill`/`forward_step` to every layer type -- a real KV-cache for the RoPE attention path (O(current_len) per generated token instead of O(current_len^2)), while MambaBlock's `forward_step` deliberately just re-runs the full op over the cached pre-block sequence (its own cost was already O(current_len), not quadratic, so there was nothing to fix there -- a documented scope decision, not an oversight).
- **Verified the cache is mathematically exact before ever touching a GPU:** `test_phase4_kvcache.py` runs entirely on CPU (no CUDA needed) and checks the cached `generate()`'s output -- both the generated token sequence AND the raw per-step logits -- against a non-cached full-recompute reference, across 5 random trials each for "transformer", "mamba", and "hybrid" kinds. Mamba/hybrid were tested with a `FakeCausalOp` stand-in (same constructor signature as `mamba_ssm.Mamba`, since the real kernel needs CUDA) that's genuinely causal, so it exercises the same cache-accumulation logic a real Mamba layer would. All passed exactly (not just "close enough").
- Measured real ARC-AGI-2 task sizes locally first (`data/arc-agi_training/evaluation_*.json`) to size the problem: at a 4,096-token (prompt+target) cap, 94.3% of training tasks and 80.0% of evaluation tasks fit -- so this phase's real numbers are reported on that 80%-coverage filtered eval subset (96/120 tasks), not the full public eval set, and that gap is stated up front, not buried.
- **Caught a real bug locally before Kaggle, via the same measure-first habit:** the leave-one-out training-example augmentation can produce examples *longer* than a task's own plain eval prompt (holding out a train pair pulls the true test pair into that example's context) -- an early task-level pre-filter under-caught this, letting a 5,230-token example slip through a 4,096 cap. Fixed by filtering leave-one-out examples individually (by their own length) rather than filtering whole tasks by their plain eval-prompt length; verified the fix locally (max example length 4,076, all <= cap) before running anything on Kaggle.
- **Data source workaround:** rather than upload the ~7MB real ARC-AGI-2 JSON files to our own Kaggle Dataset (the still-unresolved `storage.googleapis.com` block from Phase 3), added `"competition_sources": ["arc-prize-2026-arc-agi-2"]` to `notebook/phase4/kernel-metadata.json` -- Kaggle's own competition already hosts this exact data, mounted at `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/`, so no upload was needed at all.
- **Smoke test first** (2 epochs, 200 training examples, 8 eval tasks): pushed, completed cleanly in 303.9s on a real Kaggle T4 -- confirmed mamba-ssm installs and runs correctly with the new cache code, both data mounts resolve, filtering/example-building counts matched the local dry run exactly (4026 kept / 282 dropped), no crashes. This caught nothing new (the CPU test plus local filtering test had already caught the real bugs), but confirmed there wasn't a GPU-specific surprise waiting -- cheap insurance before committing to the full run, learned from Phase 3.
- **Full run**: 8 epochs (reduced from Phase 2b's 80, see below), full 4,026 filtered training examples, batch_size=16, full 96-task filtered eval set, baseline vs. +verifier (symbolic only, no TTA -- Phase 3 already showed naive TTA hurts models without rotation/reflection training augmentation, which is also true here). Completed in 4,139.5s (~69 min) -- much faster than a rough pre-run estimate of 4-5h; the KV-cache made eval cheap as hoped (baseline+verifier eval for all 3 architectures together took under 20 minutes of that total).

**Results** (full numbers: `notebook/phase4/output/phase4_results.json`, log: `notebook/phase4/output/parsed.txt`):

| Model | Params | baseline exact | baseline cell-acc | baseline shape-match |
|---|---|---|---|---|
| symbolic_only (shared) | -- | 0/139 | -- | -- |
| Transformer (RoPE) | 795,520 | 0.000 | **0.000** | **0.000** |
| Mamba | 1,022,336 | 0.000 | 0.027 | 0.036 |
| Hybrid | 1,103,872 | 0.000 | 0.061 | 0.101 |

(`with_verifier` is identical to baseline for every architecture -- the narrow curated struct+color DSL, as expected, essentially never matches real ARC-AGI-2's much more diverse transformations: 0/139 exact matches from the symbolic verifier alone, versus a perfect 1.000 on the enumerable synthetic benchmark in Phase 3. This is the expected, honestly-predicted outcome, not a bug.)

**This looks like a real reversal of the synthetic finding on first glance, but the `shape_match_rate` column reveals what's actually going on: the Transformer isn't scoring low, it's producing ZERO structurally-valid output at all** -- exactly the signature of the pre-RoPE-fix collapse mode diagnosed all the way back in the original Phase 2 pilot (`ROADMAP.md` Phase 2 section), before RoPE was even in the picture. Mamba (3.6% shape-match) and hybrid (10.1% shape-match) are already producing *some* correctly-shaped output at this same 8-epoch budget. **Most likely explanation: this is an under-training artifact, not evidence against RoPE or a genuine architecture reversal.** Two reasons to believe this over a "real reversal" reading: (1) Phase 2b's own Transformer training curve on synthetic data took until epoch 60 of 80 to reach its peak (quickcheck_cell_acc climbed 0.000 -> 0.021 -> 0.141 -> 0.191 across the run) -- it has a slow start even on the much narrower, easier synthetic task family, and this real run used only 8 epochs, an order of magnitude less runway, on a *harder and far more diverse* task distribution; (2) this exact "Mamba/hybrid produce something, Transformer produces nothing" pattern is precisely what the original Phase 2 pilot found before RoPE existed as an explanation, which we already know is a fixable training-dynamics issue, not an inherent Transformer limitation.

### 2026-09-07 — Phase 4a follow-up: longer Transformer-only run confirms under-training, not a reversal

Reran `notebook/phase4/phase4_train.py` with `ARCHITECTURES = ["transformer"]` and `LONG_RUN_EPOCHS = 40` (5x the original run's 8 epochs) -- everything else identical (same 4,026 filtered training examples, same 96-task filtered eval set). Completed in 6,572.1s (~110 min). Full numbers: `notebook/phase4/output_v3/phase4_results.json`, log: `notebook/phase4/output_v3/parsed.txt`.

Quickcheck trend across training (15-task subsample, checkpoint selection only):

| Epoch | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 39 |
|---|---|---|---|---|---|---|---|---|---|
| quickcheck_cell_acc | 0.000 | 0.000 | 0.023 | 0.004 | 0.046 | 0.034 | 0.025 | **0.053** | 0.027 |

Best checkpoint: epoch 35. Full-eval result on that checkpoint (96 real tasks, 139 test examples): **exact=0.000, cell-acc=0.079, shape-match=0.108** -- up from a flat 0.000/0.000/0.000 at 8 epochs, and now slightly ahead of hybrid's own 8-epoch numbers (cell-acc 0.061, shape-match 0.101).

**This confirms the under-training explanation directly, not a genuine architecture reversal.** Epochs 0 and 5 are still exactly 0.000 (the same dead zone the 8-epoch run never escaped), and the first sign of life appears at epoch 10 -- past the point the original 8-epoch run stopped at. This mirrors Phase 2b's own synthetic-data Transformer training curve (which took until epoch 60 of 80 to peak) and confirms the Transformer's real-data "failure" in the first Phase 4a run was simply that 8 epochs falls inside its own slow-start dead zone on a harder task distribution, not evidence that RoPE's synthetic-benchmark advantage doesn't transfer to real data.

**Still open, not closed by this run:** this isn't yet a fair 3-way comparison -- Mamba/hybrid's numbers are still from the 8-epoch run, so whether the Transformer's ~4-6x synthetic advantage also holds on real ARC-AGI-2 under a matched epoch budget for all three is unresolved. Epoch 39's quickcheck (0.027) is noticeably below epoch 35's peak (0.053) and the trend hadn't clearly plateaued -- even 40 epochs may not be this Transformer's ceiling on real data either.

### 2026-09-07 — Phase 4a closed out: fair matched-epoch 3-way real-ARC-AGI-2 comparison

Reran `notebook/phase4/phase4_train.py` with `ARCHITECTURES = ["mamba", "hybrid"]`, `LONG_RUN_EPOCHS = 40` (same budget as the Transformer follow-up) -- everything else unchanged. Completed in 4,115.6s (~69 min). Full numbers: `notebook/phase4/output_v4/phase4_results.json`, log: `notebook/phase4/output_v4/parsed.txt`.

Quickcheck trends (15-task subsample, checkpoint selection only):

| Epoch | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 39 |
|---|---|---|---|---|---|---|---|---|---|
| Mamba | 0.000 | 0.023 | 0.023 | 0.023 | 0.023 | 0.023 | 0.023 | **0.023** | 0.000 |
| Hybrid | 0.000 | 0.000 | 0.004 | 0.018 | 0.017 | 0.019 | 0.026 | 0.025 | **0.031** |

Mamba's best checkpoint: epoch 35 (flatlined at 0.023 from epoch 5 onward, then dropped to 0.000 at epoch 39 -- looks like early convergence followed by mild overfitting/degradation). Hybrid's best: epoch 39, still climbing at the end of the run (not yet plateaued).

**Final matched-epoch (40 epochs, all 3 architectures) real ARC-AGI-2 comparison** (96/120-task filtered eval set):

| Model | Params | cell-acc | shape-match |
|---|---|---|---|
| **Transformer (RoPE)** (from the earlier 40-epoch run) | 795,520 | **0.079** | **0.108** |
| Mamba | 1,022,336 | 0.047 | 0.072 |
| Hybrid | 1,103,872 | 0.044 | 0.065 |

**This closes Phase 4a's open question: the Transformer's advantage transfers to real ARC-AGI-2 data, replicating the synthetic-benchmark finding's direction, but at a much smaller margin** (~1.7x cell-acc, ~1.5x shape-match here vs. ~4-6x on the synthetic compositional benchmark). Two honest, undistinguished readings: either real ARC-AGI-2's far greater task diversity leaves less room for RoPE's specific fix (a row-counting/copy-heavy structural bug) to dominate since most real tasks aren't that narrow pattern, or 40 epochs still isn't enough runway for all three architectures (none of the three quickcheck curves had clearly plateaued by epoch 39). Good enough to state directionally in the paper with an explicit caveat about margin size; a longer run would sharpen it further but isn't required to report honestly. **This closes out Phase 4a** -- see `ROADMAP.md`'s Phase 4a section for the full write-up.

### 2026-09-08 — Phase 4b: compositional-split-severity sweep -- the paper's core mechanistic result

Extended `generator.py` with `diagonal(k)`/`split_for_severity(severity)`: the 4x4 (structural x color) combo grid decomposes into 4 disjoint mod-4-shift diagonals; holding out `severity` of them (1/2/3 of 4) leaves 12/8/4 train combos and 4/8/12 heldout combos respectively, while every primitive still appears in at least one training combo at every severity. severity=1 exactly reproduces the existing split -- verified by regenerating and hashing `data/synthetic/*.json`, byte-identical to what was already there, so Phase 2b/3's results are completely unaffected. Training task count held fixed at 600 total across severities so severity isolates compositional difficulty from data volume. Ran `test_harness.py` and `test_phase3_harness.py` after the generator change (both still pass unchanged) plus a new local check that the symbolic verifier still solves 100% of the new severity-2/3 splits (600/600, 400/400, 40/40, 600/600 exact matches across the new files) and that the tokenizer round-trips correctly on them.

Ran `notebook/phase4b/phase4b_severity_train.py` on Kaggle (Phase 2b's model/training code, reused verbatim since only the data varies): severity-2/3 data is generated on the fly inside the kernel via the already-mounted `generator.py`'s `generate_split`, with a small inlined copy of `diagonal`/`split_for_severity` (verified to match the local canonical version exactly before pushing) -- no Dataset upload needed. Completed in 9,178.3s (~153 min) for all 3 severities x 3 architectures x 80 epochs. Full numbers: `notebook/phase4b/output/phase4b_results.json`.

| Severity | Combos (train/heldout) | Transformer train_eval cell / heldout cell | relative drop |
|---|---|---|---|
| 1 (original) | 12/4 | 0.298 / 0.241 | 19.1% |
| 2 | 8/8 | 0.377 / 0.156 | 58.6% |
| 3 | 4/12 | **0.852** / **0.108** | **87.3%** |

**This is the clean mechanistic result Phase 4 was designed to produce, not just another score table.** As severity increases, the Transformer's in-distribution accuracy climbs (easier to fit 4 well-drilled combos than 12 sparser ones, same total training budget) while its held-out accuracy on genuinely novel pairings falls -- the compositional generalization gap widens monotonically and dramatically, from 19% to 87% relative drop (shape-match tells the identical story: 19.2% -> 57.9% -> 76.0%). Mamba/hybrid show the same directional trend but it isn't interpretable for either -- their own train_eval scores at severity 3 (0.124, 0.125) are far below what's needed to distinguish "failed to generalize" from "never learned its own training data," consistent with every earlier finding in this project that they're the "hasn't learned enough to be interpretable" architectures at this scale/budget.

**This effectively completes Phase 4** (both halves of the original plan: real-ARC-AGI-2 accuracy via Phase 4a, and the compositional-split-severity ablation via Phase 4b). See `ROADMAP.md`'s Phase 4b section for the full write-up.

### 2026-09-08 — Phase 5: first working Kaggle submission produced

Built `notebook/submission/submission.py`: Transformer (RoPE) only (no Mamba/hybrid, since it's both the strongest architecture in every experiment so far AND avoids needing `mamba-ssm`'s internet-requiring `pip install`, which a real scored run can't have), same ~800K-param scale as the rest of the project, `FILTER_CAP` raised to 8192 (99.9%/98.3%/100% training/public-eval/blind-test coverage, measured locally first), wall-clock-bounded training (not fixed epochs, since per-epoch cost at this new larger cap wasn't known yet), and the KV-cached `generate()` from Phase 4a (blind-test prompts run up to 6,515 tokens -- the old un-cached generate() would be far too slow here).

Smoke test (120s training, 8-task blind-test subsample, `enable_internet: false` for a genuine no-internet dry run) completed cleanly in 185.4s -- confirmed no internet needed anywhere, competition data mounts correctly, KV-cache works, submission.json format validates. Full run (6h training budget, full 240-task blind test set, still no internet) completed in 22,355.4s (~6.2h): 90 epochs, best checkpoint at epoch 76 (quickcheck_cell_acc 0.088), public-eval sanity check exact_match=0.000 (matches every real-data result in Phase 4a). **Produced a valid submission.json: 240/240 tasks, 259/259 test outputs, 0 validation problems** (2 symbolic, 174 neural, 83 fallback).

Caught and fixed one real bug via the defensive fallback working exactly as designed: one public-eval prompt alone exceeded `FILTER_CAP` (eval/blind-test prompts are never length-filtered, unlike training examples), overflowing the RoPE cache by 1 slot and raising a shape-mismatch -- caught by the existing try/except, fell back to a placeholder, no crash. Fixed the root cause anyway (margin raised from 128 to 2048 tokens) rather than just leaving it tolerated. Not yet rerun with the fix; the current submission.json is still valid and usable, the fix matters for Phase 6's actual final entry.

### 2026-09-08 — Phase 6: first competition submission made

Submitted "ARC-AGI 2 Submission - Version 2" (the run described above) to `arc-prize-2026-arc-agi-2` via Kaggle's web UI (Output tab -> Submit to Competition), after explicit user confirmation at each consequential step: a CLI `kaggle competitions submit` was tried first and correctly rejected (403 -- confirmed this is a Code Competition, entry only via the notebook editor, no submission consumed by the rejected attempt), then the actual submission was driven via Claude in Chrome browser automation with the user watching, pausing right before the final click for their go-ahead, and stopping entirely (handing back to the user) at a Kaggle identity-verification (Persona) step that needed the user's own webcam/smartphone action.

Submission is now queued as "Notebook Running" -- Kaggle privately re-runs the submitted notebook version with the hidden test set substituted in, then scores the extracted output file. Given the original run took ~6.2h, grading is expected to take a similar order of time. 1 submission used of 2 total slots this competition allows toward the final leaderboard score (submissions remaining today: 0 of 1, resets in 19h per the UI at submission time) -- **first entry into the competition, real score not yet known.**

### 2026-09-08 — First submission scored (0.00, expected); Phase 7 paper draft written

Checked the submission's score: **0.00 exact-match**, "Succeeded" status. This is expected, not a bug -- matches our own pre-submission public-eval sanity check (also 0.000) and is consistent with ARC-AGI-2 being explicitly designed to defeat small, non-test-time-trained models (comparable published approaches without TTT/vision-native representations also fall to near-zero, e.g. VARC's own 18M-param model drops from 54-60% on ARC-1 to 8-11% on ARC-2). Our own graded metrics (cell-acc ~8%, shape-match ~11% from the same model) confirm real, if shallow, learning -- exact whole-grid correctness is just a much higher bar than an 800K-param model with no test-time adaptation is likely to clear on a genuinely novel task.

Wrote `PAPER.md`: a full draft in the organizer's required structure (Abstract, Intro, Prior Work, Approach, Results, Conclusion, Reproducibility), pulling every claim and number directly from this log and `ROADMAP.md` -- no new claims introduced. Caught and fixed one real issue while writing it: the abstract initially attributed specific (fabricated, never-verified) author names to the NSA paper -- corrected to cite by arXiv ID only, consistent with how every other paper in this project is cited (verified authors only for Kallini et al., confirmed via WebSearch in an earlier session; everything else cited by nickname + arXiv ID without asserting unconfirmed names). Also fixed a section cross-reference error (pointed at a nonexistent "Section 5.3").

### 2026-09-08 — Repo prepared for open-sourcing (local half done, push pending)

Initialized git, wrote `.gitignore` (excludes the real ARC-AGI-2 competition data for redistribution-licensing safety, `__pycache__`, `.claude/`, the `kaggle_dataset/` packaging mirror, and the empty `notebook_filetest/`), a CC BY 4.0 `LICENSE`, and a `README.md`. Found and fixed a real issue before committing: `vendor/arc-dsl` had its own nested `.git` directory (from however it was originally obtained), which git was about to silently record as an empty embedded-repo reference instead of tracking its actual files -- anyone cloning the repo would have gotten an empty `vendor/arc-dsl/` directory. Removed the nested `.git` so its files are tracked normally. Scanned all staged file contents for secrets/credentials before committing (clean -- the one hit was a documentation sentence *mentioning* `kaggle.json` exists locally, not the file or its contents). Made the initial commit (88 files).

**Blocked on the actual push:** `gh` (GitHub CLI) isn't installed and I don't have push credentials for the user's GitHub account, so the remote repo creation + push needs the user to either create an empty GitHub repo and share the URL, or push it themselves.

## Next up
- [ ] Push the local repo to a GitHub remote (blocked on the user creating the repo / providing a URL, or pushing themselves -- see entry above).
- [ ] Fill in the `[repository link]` placeholders in `PAPER.md` once the repo has a public URL.
- [ ] Proofread `PAPER.md` against the actual ARC Prize Paper Track submission format/length requirements (re-verify live closer to submission, per this project's own standing practice of not trusting time-sensitive facts once time has passed) and submit it there.
- [ ] Rerun `notebook/submission/submission.py` with the MAX_SEQ_LEN margin fix for a cleaner second submission (up to 2 submissions count toward the final leaderboard score; not strictly required since the current one is already valid, but removes a known, already-handled failure mode).
- [ ] Decide on and lock in the Final Submission selection once both submission slots are used (respect the 1/day limit, don't attempt this on the deadline day itself; the real window for this is Oct 25-31 per the roadmap, well after this early first submission).
- [ ] Optional, lower priority: hybrid's own hyperparameter sweep (still deprioritized-but-not-dropped from Phase 2c) -- unlikely to change the qualitative story at this point given how consistent the "Mamba/hybrid haven't learned enough" pattern has been across Phase 2b, 2c, 4a, and 4b.
- [ ] Optional: a longer (>40 epoch) matched-budget real-ARC-AGI-2 rerun if the paper wants a sharper real-data margin than Phase 4a's 40-epoch numbers give -- not required, current numbers are honestly reportable as-is.
- [ ] Phase 6-7: final submission, paper writeup (see ROADMAP.md) -- not yet started.
- [x] Phase 3 Kaggle run — complete, see results entry above.
- [ ] Decide whether to test the TTA-hurts-generalization hypothesis (Finding 2 from Phase 3) by adding D4 rotation/reflection augmentation to training data and re-running TTA, or just report it as an honest negative result.
- [x] Phase 4a: real ARC-AGI-2 public eval set accuracy — done 2026-09-07 (see results entry above), but the Transformer's result there is ambiguous (undertrained, most likely) pending the longer follow-up run above.
- [x] Isolate exposure-bias vs. positional-encoding explanation — resolved: RoPE fixes it, scheduled sampling doesn't (see 2026-09-06 results entry above)
- [x] Re-run Phase 2 with RoPE-fixed Transformer — done, Phase 2b: Transformer now dominates Mamba/hybrid 6-7x, and shows the project's first interpretable train_eval-vs-heldout gap (see 2026-09-06 results entry above)
- [x] Phase 2b replication run 2 (seed 1) — confirmed: Transformer (RoPE) beats Mamba/hybrid ~4-6x in both seeds, and its train_eval-vs-heldout gap replicates at consistent magnitude (~23-31%). Now a robust finding, not preliminary (see 2026-09-06 results entry above)
- [x] Tune Mamba's own hyperparameters — resolved: the current shared regime already wins a 3-way sweep (see 2026-09-06 results entry above), so Mamba's weak scores are not a training-regime artifact. Phase 2b's numbers stand; no re-run needed.
- [ ] (Lower priority now that Mamba's sweep confirmed the shared regime isn't the issue) Optionally give hybrid the same sweep for full completeness before Phase 4
- [x] Read the four core papers in full (Mission: Impossible Language Models, NSA, "ARC Is a Vision Problem!", "ARC-AGI Without Pretraining") — done 2026-09-07, caught and fixed a factual error in ROADMAP's NSA summary along the way (see entry above)
- [x] Write the one-paragraph hypothesis statement for the paper's Abstract — done 2026-09-07, see `ROADMAP.md` section 4.1
- [x] Phase 3: add the NSA-style neuro-symbolic proposer+verifier layer — code complete, locally validated, and run on Kaggle 2026-09-07 (see results entry above)
