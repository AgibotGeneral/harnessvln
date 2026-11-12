"""Typed panorama access with explicit yaw lookup.

The evaluator rotates 12 frames at 30-degree intervals, shifts the last frame
to the front, and samples every third frame to produce yaws 0, 90, 180 and 270.
assert_legacy_layout checks the assumptions behind positional indexing.
Panorama provides a read-only view over the existing frame dictionaries.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .geometry import RelativeYaw

# The four canonical yaws in the order get_panorama() emits them.
CANONICAL_YAWS = (0, 90, 180, 270)

# Direction word -> yaw offset in degrees. Was duplicated as a literal dict at
# two rollout sites (backtrack replan and the grounding step-2 path).
DIRECTION_YAW = {
    'forward': 0,
    'left': 90,
    'behind': 180,
    'right': 270,
}


class Panorama:
    """Read-only, angle-keyed view over the frame list get_panorama returns."""

    __slots__ = ('frames',)

    def __init__(self, frames: Sequence[Dict[str, Any]]):
        self.frames: List[Dict[str, Any]] = list(frames)

    # -- construction -------------------------------------------------
    @classmethod
    def of(cls, frames) -> Optional["Panorama"]:
        """Wrap a frame list; return None for absent frames or an episode-done dictionary."""
        if frames is None or isinstance(frames, dict):
            return None
        return cls(frames)

    # -- sequence-ish access (legacy compatible) ----------------------
    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, index):
        return self.frames[index]

    def as_list(self) -> List[Dict[str, Any]]:
        """The exact payload legacy consumers expect."""
        return self.frames

    # -- typed lookup -------------------------------------------------
    def angles(self) -> List[int]:
        return [int(f.get('angle', -1)) for f in self.frames]

    def index_for_yaw(self, yaw) -> int:
        """Index of the frame whose tagged angle matches ``yaw``.

        Falls back to the legacy ``yaw // 90`` position only when the frames
        carry no usable ``angle`` tag, so a malformed panorama degrades to
        today's behaviour rather than raising inside a rollout.
        """
        target = RelativeYaw(float(yaw)).normalized
        for i, frame in enumerate(self.frames):
            angle = frame.get('angle')
            if angle is None:
                continue
            if float(angle) % 360.0 == target:
                return i
        return self.legacy_index(yaw)

    def legacy_index(self, yaw) -> int:
        """What the pre-harness call sites computed: ``target_angle // 90``."""
        return int(RelativeYaw(float(yaw)).normalized // 90)

    def view_for_yaw(self, yaw) -> Dict[str, Any]:
        return self.frames[self.index_for_yaw(yaw)]

    def index_for_direction(self, direction: str) -> int:
        return self.index_for_yaw(DIRECTION_YAW[str(direction).lower()])

    def view_for_direction(self, direction: str) -> Dict[str, Any]:
        return self.frames[self.index_for_direction(direction)]

    # -- invariant ----------------------------------------------------
    def legacy_layout_ok(self) -> bool:
        """True when angle lookup and ``yaw // 90`` agree on all four yaws.

        This is the assumption the legacy indexing rests on. If a future change
        to turn granularity or stride breaks it, the fixture test fails instead
        of the agent silently reading the wrong direction.
        """
        if len(self.frames) != len(CANONICAL_YAWS):
            return False
        return all(self.index_for_yaw(y) == self.legacy_index(y) for y in CANONICAL_YAWS)
