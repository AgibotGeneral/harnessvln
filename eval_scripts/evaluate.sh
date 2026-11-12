#!/usr/bin/env bash
# One entry point: evaluate.sh R2R|R2R_RAND100|RXR|HM3D_V2|HM3D_OVON|LANGMAP [config overrides]
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
source eval_scripts/lib/runtime_env.sh
resolve_python
benchmark="${1:?provide a benchmark name}"
shift
settings="$("${PYTHON}" -m vlnce_baselines.tasks "${benchmark}")"
read -r DATASET EXP_CONFIG DEFAULT_TIMEOUT_S <<< "${settings}"
source eval_scripts/lib/eval_common.sh
run_eval "$@"
