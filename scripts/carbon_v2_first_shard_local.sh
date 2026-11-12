#!/usr/bin/env bash
# Carbon-v2 local setup and first full-size shard. No cloud jobs or Actions.
# Default/readiness mode never installs packages or launches experiments.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/carbon_v2_first_shard_local.sh MODE

  --check         Check branch, clean source tree, Python version, and pinned deps.
  --install       Create .venv-carbon-v2 and install requirements.txt locally;
                  run the complete regression suite. Requires package access.
  --execute-one   Run all tests, then the ONE full-size Microsoft seed-0 shard;
                  CPU-intensive. Requires a clean branch and pinned environment.

Never dispatches GitHub Actions. Never overwrites the original v1 results.
HELP
}

mode="${1:---check}"
if [[ $# -gt 1 ]] || [[ "$mode" != "--check" && "$mode" != "--install" && "$mode" != "--execute-one" ]]; then
  usage >&2
  exit 2
fi

cd "$(dirname "$0")/.."
root="$PWD"
source_branch="$(git branch --show-current)"
if [[ "$source_branch" != "carbon-accounting-v2-validation" ]]; then
  echo "ERROR: checkout carbon-accounting-v2-validation (current: $source_branch)." >&2
  exit 3
fi

source_dirty="$(git status --porcelain --untracked-files=all -- qpeo scripts configs tests requirements.txt pyproject.toml)"
if [[ -n "$source_dirty" ]]; then
  echo "ERROR: tracked or untracked source files differ from the Git commit:" >&2
  printf '%s\n' "$source_dirty" >&2
  exit 3
fi

revision="$(git rev-parse HEAD)"
if [[ ! "$revision" =~ ^[a-f0-9]{40}$ ]]; then
  echo "ERROR: expected a 40-character source Git revision." >&2
  exit 3
fi
printf 'Clean Carbon-v2 branch at %s\n' "$revision"

venv="$root/.venv-carbon-v2"
if [[ "$mode" == "--install" ]]; then
  py="${QPEO_PYTHON:-python3.13}"
  if ! command -v "$py" >/dev/null 2>&1; then
    echo "ERROR: Python 3.13 not found; install it, or set QPEO_PYTHON to its path." >&2
    exit 3
  fi
  "$py" -c 'import sys; assert sys.version_info[:2] == (3,13), "Use Python 3.13"'
  "$py" -m venv "$venv"
  "$venv/bin/python" -m pip install --disable-pip-version-check -r requirements.txt
fi

if [[ ! -x "$venv/bin/python" ]]; then
  echo "Pinned environment missing. Run: bash scripts/carbon_v2_first_shard_local.sh --install" >&2
  exit 3
fi

"$venv/bin/python" - <<'PY'
import sys
from importlib import metadata
from pathlib import Path
required = {}
for line in Path('requirements.txt').read_text().splitlines():
    if '==' in line:
        name, version = line.split('==', 1)
        required[name.strip()] = version.strip()
errors = []
if sys.version_info[:2] != (3, 13):
    errors.append(f'Python: {sys.version.split()[0]} != 3.13.x')
for name, version in required.items():
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        actual = '(missing)'
    print(f'{name}: {actual} (required: {version})')
    if actual != version:
        errors.append(f'{name}: {actual} != {version}')
if errors:
    raise SystemExit('PIN CHECK FAILED: ' + '; '.join(errors))
print('Pinned runtime: PASS')
PY

"$venv/bin/python" -m pip check
if [[ "$mode" == "--check" ]]; then
  echo "READY: environment and source check passed; no experiment started."
  exit 0
fi

"$venv/bin/python" -m pytest -q
if [[ "$mode" == "--install" ]]; then
  echo "Pinned environment installed and regression tests passed."
  echo "To run ONE full-size shard: bash scripts/carbon_v2_first_shard_local.sh --execute-one"
  exit 0
fi

echo "Starting ONE full-size shard: Microsoft ElasticFlow, seed index 0, P=50, T=500."
"$venv/bin/python" scripts/run_carbon_v2_campaign.py run-one \
  --scenario microsoft_elasticflow_itp_cluster08 --seed-index 0 --execute

out="results/carbon_v2_campaign/${revision:0:12}"
"$venv/bin/python" scripts/audit_carbon_v2_campaign.py --input-dir "$out" --fast-audit
printf 'First-shard output directory: %s/%s\n' "$root" "$out"
echo 'IMPORTANT: This does not complete the 120-shard study or justify manuscript claims.'
