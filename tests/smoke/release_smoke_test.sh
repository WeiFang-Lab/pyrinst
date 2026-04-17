#!/usr/bin/env bash

set -euo pipefail

python -m pip uninstall -y pyrinst || true
python -m pip install --no-deps --force-reinstall dist/*.whl

pyrinst-gen-ref --help >/dev/null
pyrinst-sampling --help >/dev/null
pyrinst-fep-eval --help >/dev/null
pyrinst-optimize --help >/dev/null
