#!/usr/bin/env bash

set -euo pipefail

python -m pip uninstall -y pyrinst || true
python -m pip install --no-deps --force-reinstall dist/*.whl

SCRIPT_PATHS="$(python - <<'PY'
import os
import site
import sysconfig

paths = []

scripts_dir = sysconfig.get_path("scripts")
if scripts_dir:
    paths.append(scripts_dir)

user_base = site.getuserbase()
if user_base:
    paths.append(os.path.join(user_base, "bin"))
    paths.append(os.path.join(user_base, "Scripts"))

seen = set()
for path in paths:
    if path and path not in seen:
        seen.add(path)
        print(path)
PY
)"
while IFS= read -r path; do
  if [[ -n "${path}" && -d "${path}" ]]; then
    export PATH="${path}:$PATH"
  fi
done <<< "${SCRIPT_PATHS}"

command -v pyrinst-gen-ref >/dev/null || { echo "pyrinst-gen-ref not found on PATH=$PATH" >&2; exit 1; }
command -v pyrinst-sampling >/dev/null || { echo "pyrinst-sampling not found on PATH=$PATH" >&2; exit 1; }
command -v pyrinst-fep-eval >/dev/null || { echo "pyrinst-fep-eval not found on PATH=$PATH" >&2; exit 1; }
command -v pyrinst-optimize >/dev/null || { echo "pyrinst-optimize not found on PATH=$PATH" >&2; exit 1; }

pyrinst-gen-ref --help >/dev/null
pyrinst-sampling --help >/dev/null
pyrinst-fep-eval --help >/dev/null
pyrinst-optimize --help >/dev/null
