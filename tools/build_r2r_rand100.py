#!/usr/bin/env python
"""Install the R2R rand100 subset (Open-Nav R2R-CE, 100 val_unseen episodes).

The subset reuses R2R val_unseen episode ids and reference paths but samples
its own start headings, so it is a separate episode file rather than an id list
over vlnce_r2r.json.gz. The source is `OpenNav_R2R-CE_100_bertidx.json` from
https://github.com/YanyuanQiao/Open-Nav;
its ids must match data/datasets/stratified_samples/vlnce_r2r_rand100.json.

Usage:
    python tools/build_r2r_rand100.py --source /path/to/OpenNav_R2R-CE_100_bertidx.json
"""
import argparse
import gzip
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "data/datasets/episodes/vlnce_r2r_rand100.json.gz")
IDS = os.path.join(REPO_ROOT, "data/datasets/stratified_samples/vlnce_r2r_rand100.json")
GT = os.path.join(REPO_ROOT, "data/datasets/episodes/vlnce_r2r_gt.json.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="OpenNav_R2R-CE_100_bertidx.json")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    with open(args.source, "rb") as f:
        raw = f.read()
    episodes = json.loads(raw)["episodes"]
    ids = [str(e["episode_id"]) for e in episodes]
    expected = json.load(open(IDS))
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise SystemExit(f"{args.source}: episode ids do not match {IDS}")
    if os.path.exists(GT):
        with gzip.open(GT, "rt") as f:
            missing = set(ids) - set(json.load(f))
        if missing:
            raise SystemExit(f"{GT} lacks reference paths for {sorted(missing, key=int)}")
    else:
        print(f"warning: {GT} not found; NDTW will be unavailable")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Byte-identical payload; mtime=0 keeps the archive reproducible.
    with open(args.out, "wb") as f:
        f.write(gzip.compress(raw, mtime=0))
    print(f"wrote {args.out} ({len(ids)} episodes)")


if __name__ == "__main__":
    main()
