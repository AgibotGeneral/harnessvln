"""Navigation harness: contracts, execution trace and episode memory.

Registry and freshness checks report violations; they do not refuse actions.
TODO memory tracks stable identifiers and evidence, while graph retrieval
provides advisory context to the planner. Navigation and terminal decisions
remain in the evaluator.
"""
from .contracts import (
    EpisodeState,
    Phase,
    Receipt,
    SideEffect,
    ToolCall,
    ToolResult,
    safe_value,
)
from .geometry import (
    CameraPoint,
    ImagePixel,
    MapCell,
    NormalizedPixel,
    RelativeYaw,
    WorldPoint,
)
from .panorama import CANONICAL_YAWS, DIRECTION_YAW, Panorama
from .registry import PRIMITIVE_TOOLS, V1_TOOLS, ToolRegistry, ToolSpec, default_registry
from .retrieval import Recall, RecalledPlace, recall
from .runtime import HarnessRuntime
from .state import NAVIGATE, STOP, NavigationState
from .todo import (
    ACTIVE,
    BLOCKED,
    COMPLETED,
    PENDING,
    TODO_STATUSES,
    TodoMemory,
    is_evidence_id,
    normalize_status,
)
from .trace import EventStream
from .world_graph import (
    CONTAINS,
    EDGE_TYPES,
    FAIL_COLLISION,
    FAIL_DEAD_END,
    FAIL_GROUNDING_ERROR,
    FAIL_NO_PROGRESS,
    FAIL_STOP_REJECTED,
    FAIL_TARGET_MISIDENTIFIED,
    FAILED_FOR_SUBGOAL,
    FAILURE_TYPES,
    NAVIGABLE_TO,
    OBSERVED_FROM,
    FailureRecord,
    ObjectNode,
    PlaceNode,
    WorldGraph,
)

__all__ = [
    'ACTIVE',
    'BLOCKED',
    'CANONICAL_YAWS',
    'COMPLETED',
    'CONTAINS',
    'CameraPoint',
    'DIRECTION_YAW',
    'EDGE_TYPES',
    'EpisodeState',
    'EventStream',
    'FAILED_FOR_SUBGOAL',
    'FAILURE_TYPES',
    'FAIL_COLLISION',
    'FAIL_DEAD_END',
    'FAIL_GROUNDING_ERROR',
    'FAIL_NO_PROGRESS',
    'FAIL_STOP_REJECTED',
    'FAIL_TARGET_MISIDENTIFIED',
    'FailureRecord',
    'HarnessRuntime',
    'ImagePixel',
    'MapCell',
    'NAVIGABLE_TO',
    'NAVIGATE',
    'NavigationState',
    'NormalizedPixel',
    'OBSERVED_FROM',
    'ObjectNode',
    'PENDING',
    'PRIMITIVE_TOOLS',
    'Panorama',
    'Phase',
    'PlaceNode',
    'Recall',
    'RecalledPlace',
    'Receipt',
    'RelativeYaw',
    'STOP',
    'SideEffect',
    'TODO_STATUSES',
    'TodoMemory',
    'ToolCall',
    'ToolRegistry',
    'ToolResult',
    'ToolSpec',
    'V1_TOOLS',
    'WorldGraph',
    'WorldPoint',
    'default_registry',
    'is_evidence_id',
    'normalize_status',
    'recall',
    'safe_value',
]
