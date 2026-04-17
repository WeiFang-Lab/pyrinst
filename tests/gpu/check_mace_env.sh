#!/usr/bin/env bash

set -euo pipefail

echo "[gpu-check] Checking NVIDIA driver visibility"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu-check] nvidia-smi not found. This machine does not look GPU-ready." >&2
  exit 1
fi
nvidia-smi

echo "[gpu-check] Checking Python-side MACE runtime"
python - <<'PY'
import importlib

modules = ["torch", "ase", "mace"]
for name in modules:
    importlib.import_module(name)

import torch

print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in torch.")

print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
PY

echo "[gpu-check] MACE runtime looks available"
