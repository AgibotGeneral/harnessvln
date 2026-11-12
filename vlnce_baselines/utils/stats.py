"""Merge per-worker single-model usage into one run summary."""
import glob
import json
import os
from habitat import logger
from vlnce_baselines.usage import merge_usage


def merge_model_usage_stats(stats_dir, split='val_unseen'):
    files = sorted(glob.glob(os.path.join(stats_dir, f'model_usage_stats_{split}_r*_w*.json')))
    if not files:
        logger.info(f'No model usage files found for split {split}')
        return None
    records = []
    for path in files:
        with open(path) as stream:
            records.append(json.load(stream))
    merged = merge_usage(records)
    merged['num_processes'] = len(records)
    merged['process_stats'] = [dict(file=os.path.basename(path), stats=record)
                               for path, record in zip(files, records)]
    output = os.path.join(stats_dir, f'merged_model_usage_stats_{split}.json')
    with open(output, 'w') as stream:
        json.dump(merged, stream, indent=2)
    logger.info(f"Model {merged['model_name']}: {merged['total_calls']} calls, "
                f"{merged['total_tokens']} tokens across {len(records)} workers")
    return merged
