"""
Scoring harness matching the ARC Prize 2026 evaluation exactly:
for each task test output, 2 attempts allowed; correct if either matches the
ground truth exactly; final score = correct test-outputs / total test-outputs
(a flat average across every test output in the set, not averaged per-task first).

Also builds/validates the real submission.json shape, so the same code path
works for our synthetic eval, the real ARC-AGI-2 public eval set, and the
actual competition submission.
"""
from __future__ import annotations

Grid = list[list[int]]


def build_submission_template(challenges: dict) -> dict:
    """submission.json skeleton: attempt_1/attempt_2 default to a 1x1 [[0]] grid,
    matching the shape Kaggle's own sample_submission.json uses."""
    submission = {}
    for task_id, task in challenges.items():
        submission[task_id] = [
            {"attempt_1": [[0]], "attempt_2": [[0]]} for _ in task["test"]
        ]
    return submission


def validate_submission(submission: dict, challenges: dict) -> list[str]:
    """Returns a list of problems found (empty list == valid)."""
    problems = []
    for task_id, task in challenges.items():
        if task_id not in submission:
            problems.append(f"missing task_id {task_id}")
            continue
        entries = submission[task_id]
        if len(entries) != len(task["test"]):
            problems.append(
                f"{task_id}: expected {len(task['test'])} test outputs, got {len(entries)}"
            )
            continue
        for i, entry in enumerate(entries):
            if "attempt_1" not in entry or "attempt_2" not in entry:
                problems.append(f"{task_id}[{i}]: missing attempt_1/attempt_2")
    return problems


def score_submission(submission: dict, solutions: dict) -> dict:
    """solutions: task_id -> list of ground-truth grids (one per test output).
    Returns {"score": float, "per_task": {task_id: fraction_correct}}."""
    total, correct = 0, 0
    per_task: dict[str, float] = {}
    for task_id, gt_grids in solutions.items():
        entries = submission[task_id]
        task_total, task_correct = 0, 0
        for entry, gt in zip(entries, gt_grids):
            task_total += 1
            total += 1
            if entry["attempt_1"] == gt or entry["attempt_2"] == gt:
                task_correct += 1
                correct += 1
        per_task[task_id] = task_correct / task_total if task_total else 0.0
    return {"score": correct / total if total else 0.0, "per_task": per_task}
