#!/usr/bin/env bash

set -euo pipefail

python -m pip uninstall -y pyrinst || true
python -m pip install --no-deps --force-reinstall dist/*.whl

SCRIPT_DIR="$(python - <<'PY'
import sysconfig
print(sysconfig.get_path("scripts"))
PY
)"
export PATH="${SCRIPT_DIR}:$PATH"

command -v pyrinst-gen-ref >/dev/null
command -v pyrinst-sampling >/dev/null
command -v pyrinst-fep-eval >/dev/null
command -v pyrinst-optimize >/dev/null

pyrinst-gen-ref --help >/dev/null
pyrinst-sampling --help >/dev/null
pyrinst-fep-eval --help >/dev/null
pyrinst-optimize --help >/dev/null
