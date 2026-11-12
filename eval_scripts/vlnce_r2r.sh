#!/usr/bin/env bash
# Dataset convenience entry; configuration lives in vlnce_baselines/tasks.py.
# SUBSET=rand100 evaluates the OpenNav R2R-CE 100-episode subset instead of val_unseen.
case "${SUBSET:-}" in
    '') benchmark=vlnce_r2r ;;
    rand100) benchmark=vlnce_r2r_rand100 ;;
    *) echo "ERROR: unknown SUBSET='${SUBSET}'; expected rand100 or unset" >&2; exit 1 ;;
esac
exec bash "$(dirname "${BASH_SOURCE[0]}")/evaluate.sh" "${benchmark}" "$@"
