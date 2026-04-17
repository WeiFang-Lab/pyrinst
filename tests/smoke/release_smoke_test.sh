#!/usr/bin/env bash

set -euo pipefail

python -m pip install build
python -m build --wheel --no-isolation

python -m pip uninstall -y pyrinst || true
python -m pip install --force-reinstall dist/*.whl

pyrinst-gen-ref --help
pyrinst-sampling --help
pyrinst-fep-eval --help
pyrinst-optimize --help
