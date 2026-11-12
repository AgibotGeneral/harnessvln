#!/usr/bin/env bash
# Dataset convenience entry; configuration lives in vlnce_baselines/tasks.py.
exec bash "$(dirname "${BASH_SOURCE[0]}")/evaluate.sh" vlnce_rxr "$@"
