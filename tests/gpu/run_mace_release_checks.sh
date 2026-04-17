#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEVICE="${PYRINST_DEVICE:-cuda}"
MODEL_PATH="${PYRINST_MACE_MODEL:-}"
WORKDIR="${PYRINST_GPU_WORKDIR:-$ROOT_DIR/tests/gpu/_artifacts}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "PYRINST_MACE_MODEL is not set." >&2
  echo "Example:" >&2
  echo '  PYRINST_MACE_MODEL=/abs/path/to/model.model bash tests/gpu/run_mace_release_checks.sh' >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "MACE model file not found: ${MODEL_PATH}" >&2
  exit 1
fi

bash "$ROOT_DIR/tests/gpu/check_mace_env.sh"

mkdir -p "$WORKDIR"

echo "[mace-release] Running CPU-safe MACE unit tests"
pytest "$ROOT_DIR/tests/unittest/test_mace.py" -q

echo "[mace-release] Running CLI smoke for MACE-backed reference generation"
pushd "$WORKDIR" >/dev/null
rm -f ref.* sampled* smoke_eval.xyz

pyrinst-gen-ref "$ROOT_DIR/tests/examples/inst-fep/water.xyz" \
  -o ref \
  -P MACE \
  --model_path "$MODEL_PATH" \
  --device "$DEVICE" \
  --enable_cueq

pyrinst-sampling ref.pkl -T 300 -N 8 -n 8 -o sampled

if command -v mace_eval_configs >/dev/null 2>&1; then
  mace_eval_configs \
    --configs="sampled_0.xyz" \
    --model="$MODEL_PATH" \
    --output="smoke_eval.xyz" \
    --device="$DEVICE" \
    --enable_cueq \
    --no_forces \
    --batch_size=8
else
  echo "[mace-release] mace_eval_configs not found; skipping evaluator smoke."
fi

popd >/dev/null

echo "[mace-release] GPU/MACE release checks completed successfully"
