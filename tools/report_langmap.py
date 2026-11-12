#!/usr/bin/env python
"""Export a harnessvln LangMap round into the official LangMap result layout.

Writes <out>/log/stats_<langmap_id>.json with the id/success/spl fields that
LangMap's analyze_langmap_results.py reads, then prints overall and per-level
SR/SPL the same way it does.

Success follows the official run.py, not habitat's Success measure: an episode
succeeds if the agent ENDS within 0.25 m of a goal viewpoint, however it ended
(STOP, early stop or step limit). Habitat additionally requires STOP, so an
agent that runs out of steps next to the goal scores 0 in the raw stats but 1
here. SPL uses the annotated start-to-goal geodesic distance as the reference.
Both numbers are printed; `success`/`spl` in the export are the official ones.

Usage:
    python tools/report_langmap.py langmap-1108
    python tools/report_langmap.py langmap-1108 --out eval_data/langmap-1108/langmap_results
    python ../LangMap/analyze_langmap_results.py --eval-type single \
        --results-dir eval_data/langmap-1108/langmap_results
"""
import argparse
import glob
import gzip
import json
import math
import os
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ID_MAP = os.path.join(REPO_ROOT, "data/datasets/episodes/langmap_id_map.json")
EPISODES = os.path.join(REPO_ROOT, "data/datasets/episodes/langmap.json.gz")
SUCCESS_DISTANCE = 0.25
LEVELS = ["object", "room", "region", "instance"]
# Level mix of the full 9314-episode single-goal set (tools/build_langmap.py).
# The stratified samples draw levels equally, so the overall number is
# reweighted to this mix to stay comparable with a full-set run.
FULL_SET_MIX = {"instance": 2878, "region": 2687, "room": 2289, "object": 1460}


def finite(x):
    x = float(x or 0.0)
    return 0.0 if math.isnan(x) or math.isinf(x) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", help="exp name under eval_data/")
    ap.add_argument("--eval-root", default=os.path.join(REPO_ROOT, "eval_data"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    id_map = json.load(open(ID_MAP))
    with gzip.open(EPISODES, "rt") as f:
        ref_geo = {str(e["episode_id"]): e["info"]["geodesic_distance"] for e in json.load(f)["episodes"]}
    rows = {}
    for f in glob.glob(os.path.join(args.eval_root, args.exp, "stats", "stats_ep_ckpt_*.json")):
        rows.update(json.load(open(f)))
    if not rows:
        raise SystemExit(f"no stats under {args.eval_root}/{args.exp}/stats/")

    out = args.out or os.path.join(args.eval_root, args.exp, "langmap_results")
    os.makedirs(os.path.join(out, "log"), exist_ok=True)
    by_level = defaultdict(list)
    for ep_id, m in rows.items():
        task_id = id_map[str(ep_id)]
        dist = m.get("distance_to_goal")
        success = float(dist is not None and math.isfinite(dist) and dist < SUCCESS_DISTANCE)
        geo, path = ref_geo[str(ep_id)], finite(m.get("path_length"))
        rec = {"id": task_id, "success": success,
               "spl": success * geo / max(path, geo),
               "habitat_success": finite(m.get("success")), "habitat_spl": finite(m.get("spl")),
               "distance_to_goal": dist, "steps_taken": m.get("steps_taken")}
        with open(os.path.join(out, "log", f"stats_{task_id}.json"), "w") as f:
            json.dump(rec, f)
        by_level[task_id.split("_")[-2]].append(rec)

    allr = [r for v in by_level.values() for r in v]
    print(f"{args.exp}: {len(allr)} episodes -> {out}/log/")
    print(f"Overall  SR {100 * sum(r['success'] for r in allr) / len(allr):5.1f}%  "
          f"SPL {100 * sum(r['spl'] for r in allr) / len(allr):5.1f}%")
    print(f"(habitat, STOP required: SR {100 * sum(r['habitat_success'] for r in allr) / len(allr):5.1f}%  "
          f"SPL {100 * sum(r['habitat_spl'] for r in allr) / len(allr):5.1f}%)")
    present = [l for l in LEVELS if by_level[l]]
    w = sum(FULL_SET_MIX[l] for l in present)
    wsr = sum(FULL_SET_MIX[l] * sum(r["success"] for r in by_level[l]) / len(by_level[l]) for l in present) / w
    wspl = sum(FULL_SET_MIX[l] * sum(r["spl"] for r in by_level[l]) / len(by_level[l]) for l in present) / w
    print(f"Weighted SR {100 * wsr:5.1f}%  SPL {100 * wspl:5.1f}%  (levels reweighted to the full-set mix)")
    for lvl in LEVELS:
        if by_level[lvl]:
            v = by_level[lvl]
            print(f"  {lvl:<9} n={len(v):<5} SR {100 * sum(r['success'] for r in v) / len(v):5.1f}%  "
                  f"SPL {100 * sum(r['spl'] for r in v) / len(v):5.1f}%")


if __name__ == "__main__":
    main()
