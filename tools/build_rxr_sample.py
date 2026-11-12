#!/usr/bin/env python
"""Build the language-filtered RxR episode subset used by EPISODES=sample.

Languages default to DATASET.LANGUAGES in the task configuration.

Usage:
    python tools/build_rxr_sample.py --size 100 --seed 42
    python tools/build_rxr_sample.py --languages en-US en-IN
    python tools/build_rxr_sample.py --dry-run
"""
import argparse
import gzip
import json
import os
import random
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPISODES = os.path.join(REPO_ROOT, "data/datasets/episodes/vlnce_rxr.json.gz")
CONFIG = os.path.join(REPO_ROOT, "habitat_extensions/config/rxr.yaml")
OUT = os.path.join(REPO_ROOT, "data/datasets/stratified_samples/vlnce_rxr.json")


def languages_from_config(path):
    """Read DATASET.LANGUAGES out of the task config.

    Parsed by hand rather than with yacs so this stays runnable under the
    fixture interpreter, which has no habitat import.
    """
    with open(path) as f:
        for line in f:
            key, _, value = line.partition(":")
            if key.strip() != "LANGUAGES":
                continue
            value = value.split("#")[0].strip()
            if value.startswith("[") and value.endswith("]"):
                return [v.strip() for v in value[1:-1].split(",") if v.strip()]
    raise SystemExit(f"no DATASET.LANGUAGES found in {path}")


def stratified_sample(episodes, size, seed):
    """Pick episode IDs by cycling through scenes in descending population order.
    
    Each nonempty scene contributes one episode before any contributes another.
    """
    rng = random.Random(seed)
    cells = defaultdict(list)
    for ep in episodes:
        cells[ep["scene"]].append(ep["id"])

    order = sorted(cells, key=lambda k: (-len(cells[k]), k))
    pools = {k: rng.sample(cells[k], len(cells[k])) for k in order}

    picked = []
    while len(picked) < size:
        progressed = False
        for key in order:
            if not pools[key]:
                continue
            picked.append(pools[key].pop())
            progressed = True
            if len(picked) == size:
                break
        if not progressed:  # asked for more episodes than the split holds
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--languages", nargs="+", default=None,
                    help="override DATASET.LANGUAGES from the config")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the split without writing")
    args = ap.parse_args()

    languages = args.languages or languages_from_config(CONFIG)
    with gzip.open(EPISODES, "rt") as f:
        raw = json.load(f)["episodes"]

    allowed = set(languages)
    episodes = [
        {"id": str(e["episode_id"]),
         "scene": os.path.basename(e["scene_id"]).replace(".glb", "")}
        for e in raw
        if "*" in allowed or e["instruction"]["language"] in allowed
    ]
    print(f"languages {sorted(allowed)}: {len(raw)} -> {len(episodes)} episodes")
    if len(episodes) < args.size:
        raise SystemExit(
            f"only {len(episodes)} episodes match {sorted(allowed)}, "
            f"cannot sample {args.size}")

    sample = stratified_sample(episodes, args.size, args.seed)
    by_scene = defaultdict(int)
    scene_of = {e["id"]: e["scene"] for e in episodes}
    for ep in sample:
        by_scene[scene_of[ep]] += 1
    print(f"sample: {len(sample)} episodes over {len(by_scene)} scenes "
          f"(seed {args.seed})")
    for scene, n in sorted(by_scene.items(), key=lambda kv: -kv[1]):
        print(f"  {scene}: {n}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(sorted(sample, key=int), f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
