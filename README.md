# Architectural Inductive Bias on ARC-AGI-2

A controlled study of whether backbone architecture — Transformer, Mamba/SSM, or a Transformer+Mamba hybrid — trained *from scratch*, changes how well a model generalizes to novel compositions of grid-transformation rules it never saw during training. Built for the [ARC Prize 2026 Paper Track](https://arcprize.org/competitions/2026/paper).

**[Read the paper](PAPER.md)** · **[Full experiment log](PROGRESS.md)** · **[Project plan and design rationale](ROADMAP.md)**

## The headline result

Holding representation, training regime, and parameter count (~0.8–1.1M) fixed across three matched backbones, a RoPE Transformer generalizes to unseen (structural, color) primitive combinations 4–6x better than Mamba or a Transformer+Mamba hybrid on a controlled synthetic benchmark — and that gap **widens monotonically from a 19% relative accuracy drop to 87%** as the benchmark's compositional difficulty increases (fewer training combinations, same total training volume). The same direction of effect replicates on real ARC-AGI-2 data at a smaller margin. Full results, including two honest negative results (a neuro-symbolic verifier hitting exactly its predicted ceiling, and naive test-time augmentation actively hurting generalization), are in [PAPER.md](PAPER.md).

## Repository layout

```
generator.py              synthetic compositional-split task generator (severity-parameterized)
arc_common/
  tokenizer.py             grid <-> token-sequence encoding shared by all backbones
  scoring.py               the official ARC Prize metric + submission.json validation
  symbolic.py              NSA-style bounded program-search verifier
  tta.py                   D4 test-time-augmentation majority voting
  models.py                shared Backbone/Transformer/Mamba code, with a KV-cache for fast generation
notebook/
  phase2b/, phase2c/, ...   Kaggle training scripts for each experiment phase (see PROGRESS.md)
  phase4b/                  compositional-split-severity sweep
  submission/                the actual Kaggle competition submission notebook
test_harness.py             tokenizer/scoring self-tests (no GPU needed)
test_phase3_harness.py      symbolic verifier / TTA self-tests (no GPU needed)
test_phase4_kvcache.py      KV-cache correctness vs. a non-cached reference (no GPU needed)
```

Each `notebook/<phase>/` directory is a self-contained script pushed to Kaggle via `kaggle kernels push -p notebook/<phase>`; the shared `arc_common`/`vendor`/`generator.py` code is mounted as a private Kaggle Dataset (or, where noted in the script, inlined directly to avoid an extra upload step).

## Setup

```bash
pip install torch  # mamba-ssm additionally needs a CUDA GPU + `pip install --no-build-isolation mamba-ssm causal-conv1d`, only used on Kaggle

# Regenerate the synthetic benchmark (deterministic, fixed seeds)
python generator.py

# Real ARC-AGI-2 data (not committed to this repo -- see LICENSE) via the Kaggle API:
kaggle competitions download -c arc-prize-2026-arc-agi-2 -p data
unzip data/arc-prize-2026-arc-agi-2.zip -d data

# Run the local (no-GPU) self-tests
python test_harness.py
python test_phase3_harness.py
python test_phase4_kvcache.py
```

## Acknowledgments

Grid primitives from Michael Hodel's [arc-dsl](https://github.com/michaelhodel/arc-dsl) (MIT licensed, vendored in `vendor/arc-dsl`). Methodology adapted from Kallini et al., ["Mission: Impossible Language Models"](https://arxiv.org/abs/2401.06416) (ACL 2024).

## License

Original code, data generators, and written materials in this repository are licensed under [CC BY 4.0](LICENSE). `vendor/arc-dsl` retains its own MIT license. The ARC-AGI-2 dataset itself is not redistributed here — see Setup above.
