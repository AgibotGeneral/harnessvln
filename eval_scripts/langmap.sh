#!/usr/bin/env bash
# Dataset convenience entry; configuration lives in vlnce_baselines/tasks.py.
# Build data/datasets/episodes/langmap.json.gz first with tools/build_langmap.py.
exec bash "$(dirname "${BASH_SOURCE[0]}")/evaluate.sh" langmap "$@"
