"""Core data contracts for the Navigation Harness.

Dependency-light on purpose: no habitat, no torch, no numpy import at module
scope, so the contracts can be unit-tested from any interpreter. Everything
here is either a value object or a plain enum -- no simulator interaction.

These four dataclasses used to live in ``vlnce_baselines/harness_runtime.py``
and are unchanged in field names, ordering and ``to_dict()`` output, because
the emitted trace records are consumed by ``tools/report_rounds.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

SCHEMA_TOOL_CALL = "harness.tool_call.v1"
SCHEMA_TOOL_RESULT = "harness.tool_result.v1"
SCHEMA_RECEIPT = "harness.receipt.v1"


class SideEffect(str, Enum):
    """Tool execution effects.

    Subclasses ``str`` so a SideEffect can be written straight into a JSONL
    trace record without a custom encoder.
    """

    READ_ONLY = "READ_ONLY"
    PLANNING = "PLANNING"
    BOOKKEEPING = "BOOKKEEPING"
    WORLD_MUTATING = "WORLD_MUTATING"
    TERMINAL = "TERMINAL"

    @property
    def mutates_world(self) -> bool:
        return self in (SideEffect.WORLD_MUTATING, SideEffect.TERMINAL)

    @property
    def read_only(self) -> bool:
        """Whether a call of this class may be traced as ``read_only=True``.

        BOOKKEEPING counts as read-only with respect to the *world*: it only
        touches harness state.
        """
        return self in (SideEffect.READ_ONLY, SideEffect.PLANNING, SideEffect.BOOKKEEPING)


class Phase(str, Enum):
    """State machine labels."""

    OBSERVE = "OBSERVE"
    PLAN = "PLAN"
    GROUND = "GROUND"
    VALIDATE = "VALIDATE"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    TERMINAL = "TERMINAL"


def safe_value(value):
    """Return a JSON-friendly, compact value for Harness trace metadata.

    Moved verbatim from ``ZS_Evaluator_mp._safe_harness_value`` so the trace
    encoding lives next to the contracts it encodes. numpy is imported lazily
    and defensively: this module must stay importable without it.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [safe_value(v) for v in value[:20]]
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in list(value.items())[:40]}
    try:
        import numpy as _np
        if isinstance(value, _np.ndarray):
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, _np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


@dataclass
class ToolCall:
    call_id: str
    name: str
    step: int
    args: Dict[str, Any] = field(default_factory=dict)
    read_only: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "tool_step": int(self.step),
            "args": self.args,
            "read_only": bool(self.read_only),
            "reason": self.reason,
        }


@dataclass
class ToolResult:
    call_id: str
    name: str
    step: int
    ok: bool
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "tool_step": int(self.step),
            "ok": bool(self.ok),
            "result": self.result,
            "error": self.error,
        }


@dataclass
class Receipt:
    receipt_id: str
    call_id: str
    tool_name: str
    step_before: int
    step_after: int
    mutation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "step_before": int(self.step_before),
            "step_after": int(self.step_after),
            "mutation": self.mutation,
            "metadata": self.metadata,
        }


@dataclass
class EpisodeState:
    episode_id: Optional[str] = None
    tool_calls: int = 0
    tool_results: int = 0
    receipts: int = 0
    violations: int = 0
    terminal_emitted: bool = False
    fresh_observation_required: bool = False
    fresh_observation_reason: Optional[str] = None
    open_calls: Dict[str, ToolCall] = field(default_factory=dict)
