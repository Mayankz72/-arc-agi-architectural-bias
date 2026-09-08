"""
Phase 3: cheap test-time augmentation (TTA) via the D4 symmetry group, plus
majority-vote candidate combination. Standard, low-cost technique used by
prior top Kaggle ARC solutions. Pure Python, no torch dependency -- takes a
`predict_fn(task, test_index) -> grid | None` callback so it works with any
proposer (a trained neural model's greedy `generate`, or even the symbolic
verifier itself) and can be unit-tested locally without a GPU.

D4 has 8 elements: identity, 3 rotations, and 4 reflections (h/v mirror plus
the two diagonal mirrors). For each element g, transform the whole task by g,
run the proposer, then apply g's inverse to the prediction to bring it back
into the original input's frame -- if the proposer's errors aren't perfectly
symmetric (they generally aren't), the 8 views' predictions disagree on hard
cases and majority voting over exact-grid-match recovers the correct answer
more often than any single view.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "arc-dsl"))

from dsl import identity, rot90, rot180, rot270, hmirror, vmirror, dmirror, cmirror  # noqa: E402


def _wrap(fn):
    def wrapped(grid: list[list[int]]) -> list[list[int]]:
        g = tuple(tuple(row) for row in grid)
        return [list(row) for row in fn(g)]
    return wrapped


# (name, forward, inverse) -- rotations are each other's inverse at +/-90degrees,
# rot180 and all four reflections are each their own inverse.
D4 = [
    ("identity", _wrap(identity), _wrap(identity)),
    ("rot90", _wrap(rot90), _wrap(rot270)),
    ("rot180", _wrap(rot180), _wrap(rot180)),
    ("rot270", _wrap(rot270), _wrap(rot90)),
    ("hmirror", _wrap(hmirror), _wrap(hmirror)),
    ("vmirror", _wrap(vmirror), _wrap(vmirror)),
    ("dmirror", _wrap(dmirror), _wrap(dmirror)),
    ("cmirror", _wrap(cmirror), _wrap(cmirror)),
]


def transform_task(task: dict, fwd) -> dict:
    return {
        "train": [{"input": fwd(p["input"]), "output": fwd(p["output"])} for p in task["train"]],
        "test": [{"input": fwd(t["input"])} for t in task["test"]],
    }


def _grid_key(grid):
    return tuple(tuple(row) for row in grid)


def tta_predict(task: dict, test_index: int, predict_fn, transforms=D4):
    """Runs predict_fn under all 8 D4 views, inverts each prediction back to
    the original frame, and majority-votes by exact grid equality. Ties are
    broken in favor of the identity view's own prediction (if it's among the
    tied leaders), else the first view found. Returns (grid_or_None,
    list_of_(view_name, grid)_candidates that succeeded)."""
    candidates = []
    for name, fwd, inv in transforms:
        try:
            t_task = transform_task(task, fwd)
            pred = predict_fn(t_task, test_index)
            if pred is None:
                continue
            candidates.append((name, inv(pred)))
        except Exception:
            continue
    if not candidates:
        return None, []
    counts = Counter(_grid_key(g) for _, g in candidates)
    best_count = counts.most_common(1)[0][1]
    tied_keys = {k for k, c in counts.items() if c == best_count}
    for name, g in candidates:
        if name == "identity" and _grid_key(g) in tied_keys:
            return g, candidates
    for _, g in candidates:
        if _grid_key(g) in tied_keys:
            return g, candidates
    return candidates[0][1], candidates  # unreachable, kept for safety


def combined_predict(task: dict, test_index: int, neural_predict_fn, use_tta: bool = True, use_symbolic: bool = True):
    """The full Phase 3 pipeline: try the symbolic verifier first (it only
    returns a program-derived prediction when that program is *provably*
    consistent with every demo pair), and fall back to the neural proposer
    (optionally TTA-ensembled) when the symbolic search finds nothing.
    Returns (grid_or_None, source) where source is "symbolic" or "neural"."""
    if use_symbolic:
        from arc_common.symbolic import symbolic_predict
        sym = symbolic_predict(task, test_index)
        if sym is not None:
            return sym, "symbolic"
    if use_tta:
        pred, _ = tta_predict(task, test_index, neural_predict_fn)
    else:
        pred = neural_predict_fn(task, test_index)
    return pred, "neural"
