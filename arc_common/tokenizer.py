"""
Shared grid <-> token-sequence representation for all three backbones
(Transformer / Mamba / hybrid). Pure Python, no torch dependency, so it
runs identically locally and inside a Kaggle kernel.

Vocab (16 tokens total):
  0-9   : cell colors
  10    : ROW_SEP   -- separates rows within a single grid
  11    : GRID_IN   -- marks the start of an input grid
  12    : GRID_OUT  -- marks the start of an output grid
  13    : PAIR_END  -- marks the end of one demonstration pair
  14    : BOS
  15    : EOS

Sequence layout for one task:
  BOS
  for each train pair: GRID_IN <input tokens> GRID_OUT <output tokens> PAIR_END
  GRID_IN <test input tokens> GRID_OUT
  [ <test output tokens> EOS ]   -- only present as the training target, not the prompt

A grid's tokens are its rows flattened row-major with ROW_SEP between rows
(none trailing). Width/height are recoverable by counting cells and ROW_SEPs,
so no explicit dimension tokens are needed.
"""
from __future__ import annotations

ROW_SEP, GRID_IN, GRID_OUT, PAIR_END, BOS, EOS = 10, 11, 12, 13, 14, 15
VOCAB_SIZE = 16


def encode_grid(grid: list[list[int]]) -> list[int]:
    tokens: list[int] = []
    for i, row in enumerate(grid):
        if i > 0:
            tokens.append(ROW_SEP)
        tokens.extend(row)
    return tokens


def decode_grid(tokens: list[int]) -> list[list[int]]:
    rows: list[list[int]] = [[]]
    for t in tokens:
        if t == ROW_SEP:
            rows.append([])
        elif 0 <= t <= 9:
            rows[-1].append(t)
        else:
            raise ValueError(f"unexpected token {t} inside grid")
    if not rows[-1]:
        rows.pop()
    return rows


def encode_prompt(task: dict, test_index: int = 0) -> list[int]:
    """Encode BOS + all train pairs + the given test input, ending at GRID_OUT.
    This is what a model conditions on to predict the test output tokens."""
    tokens = [BOS]
    for pair in task["train"]:
        tokens.append(GRID_IN)
        tokens.extend(encode_grid(pair["input"]))
        tokens.append(GRID_OUT)
        tokens.extend(encode_grid(pair["output"]))
        tokens.append(PAIR_END)
    tokens.append(GRID_IN)
    tokens.extend(encode_grid(task["test"][test_index]["input"]))
    tokens.append(GRID_OUT)
    return tokens


def encode_target(output_grid: list[list[int]]) -> list[int]:
    return encode_grid(output_grid) + [EOS]


def decode_prediction(tokens: list[int]) -> list[list[int]] | None:
    """Decode a model's generated continuation after GRID_OUT back into a grid.
    Truncates at EOS. Returns None if the tokens don't form a valid grid
    (e.g. the model generated a structural token where a color was expected)."""
    if EOS in tokens:
        tokens = tokens[: tokens.index(EOS)]
    try:
        grid = decode_grid(tokens)
    except ValueError:
        return None
    if not grid or any(len(row) != len(grid[0]) for row in grid):
        return None
    return grid
