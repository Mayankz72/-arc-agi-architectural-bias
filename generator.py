"""
Synthetic ARC-style task generator for the architectural inductive-bias study.

Builds a systematic compositional-generalization split, in the spirit of SCAN/COGS
and "Mission: Impossible Language Models" (Kallini et al., ACL 2024): a small set of
STRUCTURAL primitives (geometric grid transforms) and COLOR primitives (recoloring
transforms), composed as struct_then_color. Every individual primitive appears in
several training combinations, but specific (structural, color) pairings on the
held-out diagonal are never demonstrated during training -- only at eval time.

Reuses grid primitives from Michael Hodel's arc-dsl (MIT licensed, vendor/arc-dsl).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "arc-dsl"))

from dsl import rot90, hmirror, vmirror, upscale, switch  # noqa: E402

# --- Primitive registry -----------------------------------------------------

STRUCTURAL = {
    "rot90": lambda g: rot90(g),
    "hmirror": lambda g: hmirror(g),
    "vmirror": lambda g: vmirror(g),
    "upscale2": lambda g: upscale(g, 2),
}

# color pairs chosen disjoint so each color-primitive's effect is unambiguous
COLOR = {
    "swap12": lambda g: switch(g, 1, 2),
    "swap34": lambda g: switch(g, 3, 4),
    "swap56": lambda g: switch(g, 5, 6),
    "swap78": lambda g: switch(g, 7, 8),
}

STRUCT_NAMES = list(STRUCTURAL.keys())
COLOR_NAMES = list(COLOR.keys())

# Held-out diagonal: pairing index i (structural) with index i (color) is never
# shown during training. Every primitive still appears in 3 other training combos.
HELD_OUT = list(zip(STRUCT_NAMES, COLOR_NAMES))
ALL_COMBOS = [(s, c) for s in STRUCT_NAMES for c in COLOR_NAMES]
TRAIN_COMBOS = [combo for combo in ALL_COMBOS if combo not in HELD_OUT]


# --- Phase 4b: compositional-split severity -----------------------------------
# The 4x4 (structural x color) grid decomposes into 4 disjoint "diagonals" (mod-4
# shifts): diagonal(0) is HELD_OUT above (pairing index i with index i); diagonal(k)
# pairs index i with index (i+k) mod 4. Holding out more diagonals leaves fewer
# training combos and more held-out ones, while preserving the original design's
# key property at every severity level: every individual primitive still appears
# in at least one training combo (each row/column of the grid loses exactly
# `severity` of its 4 cells, never all 4), so held-out failure is still about novel
# RECOMBINATION, not about primitives the model never saw at all.

def diagonal(k: int) -> list[tuple[str, str]]:
    return [(STRUCT_NAMES[i], COLOR_NAMES[(i + k) % 4]) for i in range(4)]


def split_for_severity(severity: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """severity = number of diagonals held out (1, 2, or 3 -- out of 4 total;
    holding out all 4 would leave zero training combos). severity=1 reproduces
    the original HELD_OUT/TRAIN_COMBOS split exactly."""
    assert 1 <= severity <= 3, "severity must leave at least one training diagonal (max 3 of 4)"
    heldout = [combo for k in range(severity) for combo in diagonal(k)]
    train = [combo for combo in ALL_COMBOS if combo not in heldout]
    return train, heldout


def random_grid(h: int, w: int, colors_in_play: tuple[int, ...], density: float = 0.35, rng: random.Random = random) -> tuple:
    """A background-0 grid with a handful of colored cells drawn from colors_in_play."""
    grid = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if rng.random() < density:
                grid[i][j] = rng.choice(colors_in_play)
    return tuple(tuple(row) for row in grid)


def make_task(struct_name: str, color_name: str, n_train: int = 3, n_test: int = 1, rng: random.Random = random) -> dict:
    """One ARC-style task: apply struct_name then color_name to random grids."""
    struct_fn = STRUCTURAL[struct_name]
    color_fn = COLOR[color_name]
    # colors this task's color-primitive actually swaps, so the effect is visible
    a, b = (int(c) for c in color_name.replace("swap", ""))
    colors_in_play = (a, b, (a * 3) % 9 + 1)  # plus one filler color unaffected by the swap

    def make_pair():
        h, w = rng.randint(3, 8), rng.randint(3, 8)
        inp = random_grid(h, w, colors_in_play, rng=rng)
        out = color_fn(struct_fn(inp))
        return {"input": [list(row) for row in inp], "output": [list(row) for row in out]}

    train = [make_pair() for _ in range(n_train)]
    test_pairs = [make_pair() for _ in range(n_test)]
    return {
        "train": train,
        "test": [{"input": p["input"]} for p in test_pairs],
        "_test_solutions": [p["output"] for p in test_pairs],  # not part of real ARC format, used for our own eval
        "_combo": [struct_name, color_name],
    }


def generate_split(combos: list[tuple[str, str]], n_tasks_per_combo: int, seed: int) -> dict:
    rng = random.Random(seed)
    challenges = {}
    solutions = {}
    combo_info = {}
    for combo_idx, (struct_name, color_name) in enumerate(combos):
        for k in range(n_tasks_per_combo):
            task_id = f"{struct_name}_{color_name}_{k:03d}"
            task = make_task(struct_name, color_name, rng=rng)
            challenges[task_id] = {"train": task["train"], "test": task["test"]}
            solutions[task_id] = task["_test_solutions"]
            combo_info[task_id] = task["_combo"]
    return {"challenges": challenges, "solutions": solutions, "combo_info": combo_info}


def main():
    out_dir = Path(__file__).parent / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Structural primitives:", STRUCT_NAMES)
    print("Color primitives:", COLOR_NAMES)
    print("Held-out (train-excluded) combos:", HELD_OUT)
    print(f"Train combos: {len(TRAIN_COMBOS)} / {len(ALL_COMBOS)} total")

    train_split = generate_split(TRAIN_COMBOS, n_tasks_per_combo=50, seed=0)
    # fresh instances of the SAME combos used in training -- measures in-distribution
    # generalization (new grids, familiar combo), the baseline to compare held-out against
    train_eval_split = generate_split(TRAIN_COMBOS, n_tasks_per_combo=10, seed=2)
    heldout_split = generate_split(HELD_OUT, n_tasks_per_combo=50, seed=1)

    for name, split in [("train", train_split), ("train_eval", train_eval_split), ("heldout", heldout_split)]:
        (out_dir / f"{name}_challenges.json").write_text(json.dumps(split["challenges"]))
        (out_dir / f"{name}_solutions.json").write_text(json.dumps(split["solutions"]))
        (out_dir / f"{name}_combo_info.json").write_text(json.dumps(split["combo_info"]))
        print(f"{name}: {len(split['challenges'])} tasks -> {out_dir / (name + '_challenges.json')}")


def main_severity_sweep():
    """Phase 4b: generates severity 2 and 3 datasets (severity 1 is the original
    data/synthetic/ above, left untouched). Training task COUNT is held fixed at
    600 total across severities (tasks_per_combo = 600 // n_train_combos) so
    "severity" isolates compositional difficulty from total training-data volume;
    heldout/train_eval keep the original per-combo densities (50 and 10
    respectively), so their totals naturally grow with severity -- more held-out
    combos need more heldout tasks to sample precisely, which is fine for an eval
    set."""
    for severity in [2, 3]:
        train_combos, heldout_combos = split_for_severity(severity)
        out_dir = Path(__file__).parent / "data" / f"synthetic_sev{severity}"
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks_per_train_combo = 600 // len(train_combos)
        print(f"\n=== severity {severity}: {len(train_combos)} train combos, "
              f"{len(heldout_combos)} heldout combos, {tasks_per_train_combo} tasks/train-combo "
              f"({tasks_per_train_combo * len(train_combos)} total train tasks) ===")
        print("train combos:", train_combos)
        print("heldout combos:", heldout_combos)

        train_split = generate_split(train_combos, n_tasks_per_combo=tasks_per_train_combo, seed=0)
        train_eval_split = generate_split(train_combos, n_tasks_per_combo=10, seed=2)
        heldout_split = generate_split(heldout_combos, n_tasks_per_combo=50, seed=1)

        for name, split in [("train", train_split), ("train_eval", train_eval_split), ("heldout", heldout_split)]:
            (out_dir / f"{name}_challenges.json").write_text(json.dumps(split["challenges"]))
            (out_dir / f"{name}_solutions.json").write_text(json.dumps(split["solutions"]))
            (out_dir / f"{name}_combo_info.json").write_text(json.dumps(split["combo_info"]))
            print(f"{name}: {len(split['challenges'])} tasks -> {out_dir / (name + '_challenges.json')}")


if __name__ == "__main__":
    main()
    main_severity_sweep()
