"""Tool metadata and validation for navigation traces.

Each tool declares its side effects, required arguments and simulator-step
usage. Validation reports unknown tools and contract violations; the runtime
records these findings without refusing the action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .contracts import SideEffect


@dataclass(frozen=True)
class ToolSpec:
    name: str
    side_effect: SideEffect
    description: str = ""
    required_args: Tuple[str, ...] = ()
    # Consumes simulator steps even when the tool does not change task state.
    step_consuming: bool = False
    # Physically moves the agent even if its *information* side effect is
    # READ_ONLY. A panorama rotates the robot and produces action receipts
    # even though its information-level operation only observes the scene.
    physical_motion: bool = False
    # Requires a passed verify_stop before execution.
    requires_stop_evidence: bool = False

    @property
    def mutates_world(self) -> bool:
        return self.side_effect.mutates_world

    def missing_args(self, args: Optional[Dict]) -> List[str]:
        provided = set((args or {}).keys())
        return [a for a in self.required_args if a not in provided]


@dataclass
class ToolRegistry:
    tools: Dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self.tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self.tools[spec.name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self.tools.get(name)

    def has(self, name: str) -> bool:
        return name in self.tools

    def names(self) -> List[str]:
        return sorted(self.tools)

    def names_by_side_effect(self, side_effect: SideEffect) -> List[str]:
        return sorted(n for n, s in self.tools.items() if s.side_effect == side_effect)

    def validate(self, name: str, args: Optional[Dict] = None,
                 read_only: Optional[bool] = None) -> List[Tuple[str, str]]:
        """Return a list of ``(kind, message)`` problems; empty means valid.

        Three checks, each mapping to a concrete past failure mode:
        unregistered name (typo'd tool goes unnoticed), missing required arg
        (a move with no pixel), and a side-effect mismatch (a world mutation
        traced as ``read_only=True``, which would let it skip the
        fresh-observation obligation).
        """
        problems: List[Tuple[str, str]] = []
        spec = self.get(name)
        if spec is None:
            problems.append(('unregistered_tool', f"tool '{name}' is not in the registry"))
            return problems
        missing = spec.missing_args(args)
        if missing:
            problems.append(('missing_args', f"tool '{name}' missing args: {', '.join(missing)}"))
        if read_only is not None:
            if bool(read_only) and spec.mutates_world:
                problems.append((
                    'side_effect_mismatch',
                    f"tool '{name}' is {spec.side_effect.value} but was called read_only",
                ))
            if not bool(read_only) and not (spec.mutates_world or spec.physical_motion):
                problems.append((
                    'side_effect_mismatch',
                    f"tool '{name}' is {spec.side_effect.value} but was called as a mutation",
                ))
        return problems


# --------------------------------------------------------------------------
# The v1 tool set plus the primitives the current evaluator actually
# traces. Both are registered on purpose: the design-level tools name intent,
# the primitives name what habitat is asked to do. Dropping either would make
# some existing trace record "unregistered" and bury the real violations in
# noise.
# --------------------------------------------------------------------------
V1_TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec('observe_panorama', SideEffect.READ_ONLY,
             'Standardised panorama capture; consumes sim steps.',
             step_consuming=True, physical_motion=True),
    ToolSpec('query_depth', SideEffect.READ_ONLY,
             'Metric depth and local depth statistics at a view/pixel.'),
    ToolSpec('recall_place', SideEffect.READ_ONLY,
             'Read a cached, already-visited place of this episode.'),
    ToolSpec('plan_navigation', SideEffect.PLANNING,
             'planner planner turn: choose the next high-level action.'),
    ToolSpec('ground_target', SideEffect.PLANNING,
             'grounding grounding: produce bbox/pixel candidates for an intent.'),
    ToolSpec('project_target', SideEffect.PLANNING,
             'Back-project a grounded pixel to a world/map goal.'),
    ToolSpec('verify_stop', SideEffect.PLANNING,
             'Geometric gate AND task-specific visual confirmation.'),
    ToolSpec('move_to_pixel', SideEffect.WORLD_MUTATING,
             'Depth back-projection, reachability check, safe motion.'),
    ToolSpec('backtrack', SideEffect.WORLD_MUTATING,
             'Return to a waypoint; forces fresh observe before replan.'),
    ToolSpec('stop', SideEffect.TERMINAL,
             'Terminate the episode, referencing passed stop evidence.',
             requires_stop_evidence=True),
)

# Habitat action primitives, as named by ZS_Evaluator_mp._action_tool_name.
PRIMITIVE_TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec('move', SideEffect.WORLD_MUTATING, 'MOVE_FORWARD primitive.',
             step_consuming=True),
    ToolSpec('turn', SideEffect.WORLD_MUTATING, 'TURN_LEFT / TURN_RIGHT primitive.',
             step_consuming=True),
    ToolSpec('action', SideEffect.WORLD_MUTATING, 'Unclassified habitat action primitive.',
             step_consuming=True),
)


def default_registry(extra: Iterable[ToolSpec] = ()) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in V1_TOOLS:
        registry.register(spec)
    for spec in PRIMITIVE_TOOLS:
        registry.register(spec)
    for spec in extra:
        registry.register(spec)
    return registry
