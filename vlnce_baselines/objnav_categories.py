"""Goal-category vocabulary and confusable sets for ObjectNav negative constraints.

The negative-constraint line used to be built against a hard-coded 6-category
list (the HM3D ObjectNav set). On MP3D's 21-category set that meant 15 of 21
categories never matched, so 72% of episodes ran with an empty constraint line --
and the 6 that did match were warned off HM3D classes rather than the objects
they actually get confused with. The vocabulary here is read from the episode
dataset's own ``category_to_task_category_id`` instead, so the constraint always
reflects the set being evaluated.

Two regimes, because "every other goal class" only works as a confusable set
when the vocabulary is small:

* Small vocabularies (HM3D/OVON-6): list every other goal class. This is the
  same set the hm3d rounds on record were scored against, so those numbers stay
  comparable (the order is now alphabetical rather than the old hand-written
  one -- same names, different sequence in the sentence).
* Large vocabularies (MP3D-21, OVON-49): listing every other class is 20-48
  items of prompt noise. Use the curated per-category set below, and emit no
  constraint at all for a category we have no entry for -- no worse than the
  empty line those categories got before.
"""

import gzip
import json
import os

# Above this many goal categories, "every other category" stops being a usable
# confusable set and CONFUSABLES takes over. HM3D/OVON-6 sit at 6, MP3D at 21.
SMALL_VOCAB_MAX = 8

# What each MP3D goal category actually gets mistaken for. Entries are free text,
# not restricted to the goal vocabulary: the failures worth suppressing are
# "stopped at a mirror thinking it was a picture" and "stopped at a dresser
# thinking it was a cabinet", and mirror/dresser are not goal classes.
CONFUSABLES = {
    "bathtub": ["shower", "sink", "toilet", "basin"],
    "bed": ["sofa", "cushion", "seating", "mattress", "bench"],
    "cabinet": ["chest_of_drawers", "counter", "cupboard", "shelf", "kitchen unit"],
    "chair": ["stool", "seating", "sofa", "bench", "armchair"],
    "chest_of_drawers": ["cabinet", "counter", "table", "nightstand", "dresser"],
    "clothes": ["towel", "curtain", "cushion", "bag", "laundry basket"],
    "counter": ["table", "cabinet", "chest_of_drawers", "sink", "kitchen island"],
    "cushion": ["pillow", "sofa", "seating", "bed", "towel"],
    "fireplace": ["tv_monitor", "picture", "cabinet", "oven", "heater"],
    "gym_equipment": ["chair", "stool", "table", "bicycle", "machinery"],
    "picture": ["tv_monitor", "mirror", "window", "poster", "whiteboard"],
    "plant": ["flower vase", "tree", "shrub", "decoration"],
    "seating": ["sofa", "chair", "stool", "bench", "cushion"],
    "shower": ["bathtub", "sink", "curtain", "glass door", "toilet"],
    "sink": ["counter", "bathtub", "toilet", "basin", "shower"],
    "sofa": ["seating", "chair", "bed", "cushion", "bench"],
    "stool": ["chair", "seating", "table", "ottoman", "bench"],
    "table": ["counter", "desk", "cabinet", "chest_of_drawers", "nightstand"],
    "toilet": ["bathtub", "sink", "bidet", "shower"],
    "towel": ["clothes", "curtain", "cushion", "bath mat", "blanket"],
    "tv_monitor": ["picture", "mirror", "laptop screen", "whiteboard", "window"],
}


def load_vocabulary(data_path):
    """Goal categories declared by an episode dataset, sorted.

    Returns [] when the file is missing or carries no category table (the VLN
    sets do not), which switches negative constraints off rather than
    guessing a vocabulary.
    """
    if not data_path or not os.path.exists(data_path):
        return []
    try:
        opener = gzip.open if data_path.endswith(".gz") else open
        with opener(data_path, "rt", encoding="utf-8") as f:
            table = json.load(f).get("category_to_task_category_id") or {}
        return sorted(table)
    except Exception as e:  # a malformed set must not take the run down
        # Imported here, not at module scope: everything else in this file is
        # pure data, and keeping habitat out of the import graph lets the
        # fixture tests exercise it without a habitat interpreter.
        from habitat import logger
        logger.info(f"objnav vocabulary: could not read {data_path} ({e}); "
                    f"negative constraints disabled")
        return []


def resolve_target(instruction, vocabulary):
    """The goal category named in `instruction`, or None.

    Longest name first so "chest_of_drawers" is not shadowed by a vocabulary
    that also contains "chest", and "tv_monitor" matches its spelled-out form.
    """
    if not instruction or not vocabulary:
        return None
    lowered = instruction.lower()
    for category in sorted(vocabulary, key=len, reverse=True):
        if category in lowered or category.replace("_", " ") in lowered:
            return category
    return None


def negative_constraints(instruction, vocabulary):
    """The `{negative_constraints}` prompt line for this instruction, or "".

    Empty when no goal category is named, or when a large vocabulary has no
    curated confusable set for the one that is.
    """
    target = resolve_target(instruction, vocabulary)
    if target is None:
        return ""

    if len(vocabulary) <= SMALL_VOCAB_MAX:
        others = [c for c in vocabulary if c != target]
    else:
        others = CONFUSABLES.get(target, [])
    if not others:
        return ""

    pretty = lambda s: s.replace("_", " ")
    return (f"- When looking for {pretty(target)}, do not confuse it with "
            f"other objects: {', '.join(pretty(o) for o in others)}.")
