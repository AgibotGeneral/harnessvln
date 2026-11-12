#!/usr/bin/env python
"""Print the VLN-literature metrics (NE / OSR / NDTW / SR / SPL / SDTW) for VLN rounds.

`report_rounds.py` compares two rounds on SR/SPL/Oracle-SR; this reports the
metrics VLN papers quote. Several exp names given to one flag are merged (the
first listed wins on duplicate episode ids, so list a resumed round first). RxR
rounds are filtered on instruction.language.

Only stdlib is used, so this runs under any interpreter (no habitat needed).

Usage:
    python tools/report_vln_ne.py --r2r r2r-eval
    python tools/report_vln_ne.py --r2r r2r-eval-b r2r-eval --rxr rxr-eval --rxr-lang en-US
"""
import argparse
import glob
import gzip
import json
import math
import os
import statistics
from collections import Counter

EVAL_DATA_ROOT = os.environ.get("EVAL_DATA_ROOT", "eval_data")
RXR_EPISODES = "data/datasets/episodes/vlnce_rxr.json.gz"

# (key, label, is_percent, higher_is_better)
METRICS = [
    ("distance_to_goal", "NE (m)", False, False),
    ("oracle_success", "OSR", True, True),
    ("ndtw", "NDTW", True, True),
    ("success", "SR", True, True),
    ("spl", "SPL", True, True),
    ("sdtw", "SDTW", True, True),
]


def load_round(exp):
    """Merge one round's per-worker stats. Truncated files (worker killed
    mid-write) are reported rather than silently shrinking the denominator."""
    eps = {}
    pattern = os.path.join(EVAL_DATA_ROOT, exp, "stats", "stats_ep_ckpt_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  !! no stats files under {pattern}")
    for path in files:
        try:
            with open(path) as f:
                eps.update(json.load(f))
        except Exception as e:
            print(f"  !! {path}: {e}")
    return eps


def merge_rounds(exps):
    """First round listed wins on duplicate episode ids (it is the newer one)."""
    merged = {}
    for exp in exps:
        for ep, m in load_round(exp).items():
            merged.setdefault(ep, m)
    return merged


def rxr_languages():
    if not os.path.exists(RXR_EPISODES):
        print(f"  !! {RXR_EPISODES} missing -- cannot filter RxR by language")
        return None
    with gzip.open(RXR_EPISODES) as f:
        episodes = json.load(f)["episodes"]
    return {str(e["episode_id"]): e["instruction"]["language"] for e in episodes}


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def report(name, eps):
    print(f"\n=== {name}  (n={len(eps)}) ===")
    if not eps:
        return
    for key, label, is_pct, _ in METRICS:
        vals = [m[key] for m in eps.values() if _finite(m.get(key))]
        if not vals:
            print(f"  {label:8s} n/a")
            continue
        mean, med = sum(vals) / len(vals), statistics.median(vals)
        if is_pct:
            mean, med = mean * 100, med * 100
            body = f"mean={mean:5.1f}%  median={med:5.1f}%"
        else:
            body = f"mean={mean:5.2f}   median={med:5.2f} "
        # A shrinking denominator means non-finite values -- the navmesh-island
        # signature (distance_to_goal=inf, spl=nan). Say so rather than letting
        # the mean quietly describe a different episode set than n.
        drop = len(eps) - len(vals)
        note = f"  [{drop} non-finite excluded, n={len(vals)}]" if drop else ""
        print(f"  {label:8s} {body}{note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2r", nargs="+", default=[], help="R2R / R2R rand100 / LangMap exp names")
    ap.add_argument("--rxr", nargs="+", default=[], help="RxR exp names")
    ap.add_argument("--rxr-lang", default="en-US",
                    help="instruction.language to keep; 'all' disables filtering")
    args = ap.parse_args()
    if not args.r2r and not args.rxr:
        ap.error("give at least one of --r2r / --rxr")

    if args.r2r:
        report(f"R2R-style round  {' + '.join(args.r2r)}", merge_rounds(args.r2r))
    if not args.rxr:
        return

    raw = merge_rounds(args.rxr)
    lang = rxr_languages()
    label = " + ".join(args.rxr)
    if lang is None or args.rxr_lang == "all":
        report(f"VLN-CE RxR (all languages)  {label}", raw)
        return
    print("\nRxR merged stats by language:",
          dict(Counter(lang.get(k, "?") for k in raw)))
    keep = {k: v for k, v in raw.items() if lang.get(k) == args.rxr_lang}
    report(f"VLN-CE RxR ({args.rxr_lang})  {label}", keep)
    other = {k: v for k, v in raw.items() if lang.get(k) != args.rxr_lang}
    if other:
        report(f"VLN-CE RxR (NOT {args.rxr_lang} -- do not report as {args.rxr_lang})",
               other)


if __name__ == "__main__":
    main()
