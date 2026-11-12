"""Explicit navigation state for one episode.

Target timeout, target arrival and rejected STOP use separate transitions.
A timeout preserves backtracking and planner intent; arrival resets both.
A rejected STOP retains the target while requesting a fresh observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

NAVIGATE = 'NAVIGATE'
STOP = 'STOP'


@dataclass
class NavigationState:
    """The live navigation state of one episode."""

    # --- stop intent -------------------------------------------------
    going_to_stop: bool = False
    # --- observation / decision progress ----------------------------
    panorama_got: bool = False
    navigate_or_not: bool = False
    # --- backtracking ------------------------------------------------
    is_backtracking: bool = False
    #: True only while returning to an old waypoint *before* re-deciding.
    backtracking_only: bool = False
    # --- planner decision -------------------------------------------------
    planned_action: str = NAVIGATE
    # --- current target (map cell coordinates) ----------------------
    target_map_x: Optional[float] = None
    target_map_y: Optional[float] = None
    target_set_step: Optional[int] = None
    #: Controller goal, np.array([target_map_y, target_map_x]).
    waypoint: Any = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def has_target(self) -> bool:
        return self.target_map_x is not None and self.target_map_y is not None

    @property
    def wants_stop(self) -> bool:
        return self.planned_action == STOP

    def steps_since_target_set(self, step: int) -> Optional[int]:
        if self.target_set_step is None:
            return None
        return int(step) - int(self.target_set_step)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def set_target(self, target_map_x, target_map_y, step: Optional[int] = None) -> None:
        self.target_map_x = target_map_x
        self.target_map_y = target_map_y
        if step is not None:
            self.target_set_step = int(step)

    def _end_target_pursuit(self) -> None:
        """The five fields both target-exit paths clear.

        Private on purpose: callers must pick the named transition that matches
        their situation, because the two paths do *not* clear the same set.
        """
        self.panorama_got = False
        self.navigate_or_not = False
        self.target_map_x = None
        self.target_map_y = None
        self.target_set_step = None
        self.backtracking_only = False

    def on_target_timeout(self) -> None:
        """Drop a timed-out target and request replanning.
        
        Preserve ``is_backtracking`` and ``planned_action`` across the timeout.
        Unlike :meth:`on_target_reached`, this transition does not reset either field.
        """
        self._end_target_pursuit()

    def on_target_reached(self) -> None:
        """Target reached: drop it and return to the normal decision flow."""
        self._end_target_pursuit()
        self.is_backtracking = False
        self.planned_action = NAVIGATE

    def on_stop_rejected(self) -> None:
        """STOP was rejected (by the geometric gate or the double check).

        Note what this does *not* touch: the current target, ``target_set_step``,
        ``is_backtracking`` and ``backtracking_only`` all survive a rejection,
        because the agent is expected to keep navigating toward the same target
        and try again from closer. Only the stop intent and the observation
        progress are rolled back.
        """
        self.going_to_stop = False
        self.navigate_or_not = False
        # Cleared so the next round re-enters Step 1 and takes a fresh panorama.
        self.panorama_got = False
        self.planned_action = NAVIGATE

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """JSON-friendly view, for trace records."""
        return {
            'going_to_stop': bool(self.going_to_stop),
            'panorama_got': bool(self.panorama_got),
            'navigate_or_not': bool(self.navigate_or_not),
            'is_backtracking': bool(self.is_backtracking),
            'backtracking_only': bool(self.backtracking_only),
            'planned_action': self.planned_action,
            'has_target': self.has_target,
            'target_set_step': self.target_set_step,
        }
