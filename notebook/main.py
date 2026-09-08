import subprocess
import sys

import torch

print("=== Environment check ===")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

print("\n=== Installing mamba-ssm ===")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-build-isolation", "mamba-ssm", "causal-conv1d"],
    capture_output=True, text=True,
)
print(result.stdout[-4000:])
print(result.stderr[-4000:])

print("\n=== Importing mamba-ssm ===")
try:
    from mamba_ssm import Mamba

    model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to("cuda")
    x = torch.randn(2, 10, 64).to("cuda")
    y = model(x)
    print("mamba-ssm works. Output shape:", y.shape)
except Exception as e:
    print("mamba-ssm FAILED:", repr(e))

print("\n=== ARC-AGI-2 data check ===")
import json
import os

print("/kaggle/input contents:", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "MISSING")
data_dir = "/kaggle/input/arc-prize-2026-arc-agi-2"
if os.path.exists(data_dir):
    print("Files:", os.listdir(data_dir))
    with open(os.path.join(data_dir, "arc-agi_training_challenges.json")) as f:
        train = json.load(f)
    print("Training tasks:", len(train))
else:
    print("Competition data not attached at", data_dir)
