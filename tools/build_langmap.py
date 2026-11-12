#!/usr/bin/env python
"""Convert LangMap single-goal annotations into a VLN-CE-v1 episode file.

LangMap ships one json.gz per HM3D val scene. The instruction text, goal
viewpoints and blacklist follow LangMapDatasetV1 in the official repo
(VLN_CE/habitat_extensions/task.py), so the episode set matches the paper's
single-goal protocol. Episode ids are renumbered 0..N-1 because run_mp.py's
resume parser only reads numeric `ep<id>:` tags; the LangMap task id
(e.g. 00800-TEEsavR23oF_instance_3) is kept in info.langmap_id and in the
id map written next to the episode file.

Usage:
    python tools/build_langmap.py
    python tools/build_langmap.py --langmap-repo ../LangMap --detailed
"""
import argparse
import ast
import gzip
import json
import math
import os
import random
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS = os.path.join(REPO_ROOT, "data/datasets/langmap/annotations")
OUT = os.path.join(REPO_ROOT, "data/datasets/episodes/langmap.json.gz")
SAMPLE_OUT = os.path.join(REPO_ROOT, "data/datasets/stratified_samples/langmap.json")
SCENE_PREFIX = "hm3d/hm3d_v0.2/val"
LEVELS = ["instance", "region", "room", "object"]


def load_blacklist(langmap_repo):
    """Read black_task_ids_ out of the official task.py without importing habitat."""
    path = os.path.join(langmap_repo, "VLN_CE/habitat_extensions/task.py")
    src = open(path).read()
    start = src.index("black_task_ids_")
    start = src.index("[", start)
    end = src.index("]", start) + 1
    return set(ast.literal_eval(src[start:end]))


def instruction_for(ep, level, goals_by_id, regions, detailed):
    if level in ("region", "room", "object"):
        category = ep["object_category"]
    else:
        category = goals_by_id[ep["target_object_ids"][0]].get("object_category", "object")
    if level == "object":
        return f"Find the {category}."
    if level == "room":
        room = ep.get("room_name", "room")
        text = f"Find the {category} in the {room}"
        if "tie:" in room.lower():
            text += ", 'Tie: A & B' means the area spans both rooms A and B"
        return text
    if level == "region":
        region = regions.get(ep.get("region_id", ""))
        if region is None:
            return f"Find the {category} in the target region"
        region_category = region.get("region_category", "region").lower()
        desc = region.get("detailed_description" if detailed else "concise_description", "")
        text = f"Find the {category} in the {region_category} that has {desc}"
        if "tie:" in region_category or "tie:" in desc.lower():
            text += ", 'Tie: A & B' means the area spans both rooms A and B"
        return text
    goal = goals_by_id.get(ep.get("instance_id", ""))
    if goal is None:
        return "Find the target instance"
    key = "annot_unique_detailed_description" if detailed else "annot_unique_concise_description"
    return f"Find the {goal.get(key, 'object')}"


def valid_start(ep):
    pos, rot = ep.get("start_position", []), ep.get("start_rotation", [])
    if len(pos) != 3 or len(rot) != 4:
        return False
    if any(math.isnan(x) or math.isinf(x) for x in pos + rot):
        return False
    return abs(math.sqrt(sum(x * x for x in rot)) - 1.0) <= 0.1


def build(annotations, blacklist, detailed):
    episodes, skipped = [], Counter()
    for fname in sorted(f for f in os.listdir(annotations) if f.endswith(".json.gz")):
        scene = fname.split(".")[0]
        glb = scene.split("-", 1)[1] if "-" in scene else scene
        with gzip.open(os.path.join(annotations, fname), "rt", encoding="utf-8") as f:
            data = json.load(f)
        goals_by_id = {g["object_id"]: g for g in data["goals"]}
        regions = data["region_annotation"]
        for level in LEVELS:
            for ep in data.get(f"episodes_by_{level}_level", []):
                task_id = f"{scene}_{level}_{ep['episode_id']}"
                if task_id in blacklist:
                    skipped["blacklist"] += 1
                    continue
                if not valid_start(ep):
                    skipped["bad_start"] += 1
                    continue
                goals = []
                for obj_id in ep.get("target_object_ids", []):
                    goal = goals_by_id.get(obj_id)
                    if goal is None:
                        continue
                    vps = [vp["agent_state"]["position"] for vp in goal.get("view_points", [])
                           if vp.get("agent_state", {}).get("position")]
                    for p in vps or [goal["position"]]:
                        goals.append({"position": p, "radius": 0.25})
                if not goals:
                    skipped["no_goals"] += 1
                    continue
                episodes.append({
                    "episode_id": len(episodes),
                    "trajectory_id": len(episodes),
                    "scene_id": f"{SCENE_PREFIX}/{scene}/{glb}.basis.glb",
                    "start_position": ep["start_position"],
                    "start_rotation": ep["start_rotation"],
                    "goals": goals,
                    "info": {
                        "langmap_id": task_id,
                        "navigation_type": level,
                        "scene_name": scene,
                        "geodesic_distance": ep["min_dist"][0]["geodesic_distance"],
                        "target_object_ids": ep["target_object_ids"],
                    },
                    "instruction": {
                        "instruction_text": instruction_for(ep, level, goals_by_id, regions, detailed),
                        "instruction_tokens": [],
                    },
                })
    return episodes, skipped


def stratified_sample(episodes, per_level, seed):
    """`per_level` ids for each goal level, round-robin over scenes within a level.

    Equal counts per level keep the per-level SR estimates equally tight (the
    full set is 31/29/25/16% instance/region/room/object); report_langmap.py
    reweights the overall number back to the full-set mix. The draw is a
    deterministic greedy walk, so a larger per_level with the same seed is a
    superset of a smaller one and a finished small round can be extended.
    """
    rng = random.Random(seed)
    out = []
    for level in LEVELS:
        cells = defaultdict(list)
        for ep in episodes:
            if ep["info"]["navigation_type"] == level:
                cells[ep["info"]["scene_name"]].append(str(ep["episode_id"]))
        for ids in cells.values():
            ids.sort(key=int)
            rng.shuffle(ids)
        scenes = sorted(cells)
        rng.shuffle(scenes)
        picked = []
        while len(picked) < per_level and any(cells.values()):
            for sc in scenes:
                if cells[sc] and len(picked) < per_level:
                    picked.append(cells[sc].pop())
        out.extend(picked)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=ANNOTATIONS)
    ap.add_argument("--langmap-repo", default=os.path.join(REPO_ROOT, "../LangMap"))
    ap.add_argument("--detailed", action="store_true",
                    help="use detailed region/instance descriptions (paper reports concise as the main result)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--per-level", type=int, nargs="+", default=[50, 100],
                    help="sample sizes per goal level; the first is written to langmap.json (EPISODES=sample)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    blacklist = load_blacklist(args.langmap_repo)
    episodes, skipped = build(args.annotations, blacklist, args.detailed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump({"episodes": episodes, "instruction_vocab": {"word_list": []}}, f)
    id_map = {str(ep["episode_id"]): ep["info"]["langmap_id"] for ep in episodes}
    id_map_path = args.out.replace(".json.gz", "_id_map.json")
    with open(id_map_path, "w") as f:
        json.dump(id_map, f, indent=0)

    os.makedirs(os.path.dirname(SAMPLE_OUT), exist_ok=True)
    sample_paths = []
    for i, n in enumerate(args.per_level):
        sample = stratified_sample(episodes, n, args.seed)
        paths = [SAMPLE_OUT.replace(".json", f"_L{n}.json")] + ([SAMPLE_OUT] if i == 0 else [])
        for path in paths:
            with open(path, "w") as f:
                json.dump(sample, f)
            sample_paths.append(f"{path} ({len(sample)} ids)")

    levels = Counter(ep["info"]["navigation_type"] for ep in episodes)
    scenes = {ep["info"]["scene_name"] for ep in episodes}
    print(f"blacklist: {len(blacklist)} ids; skipped: {dict(skipped)}")
    print(f"{len(episodes)} episodes over {len(scenes)} scenes: {dict(levels)}")
    print(f"wrote {args.out}\n      {id_map_path}")
    for p in sample_paths:
        print(f"      {p}")


if __name__ == "__main__":
    main()
