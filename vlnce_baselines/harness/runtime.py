"""Harness runtime with append-only traces and registry-checked tool calls.

Unknown tools, mismatched side effects and missing fresh observations produce
HARNESS_VIOLATION records. These checks report violations without refusing actions.
Episode observations and action receipts provide IDs for evidence validation.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .contracts import (
    EpisodeState,
    Phase,
    Receipt,
    SideEffect,
    ToolCall,
    ToolResult,
)
from .registry import ToolRegistry, default_registry


class HarnessRuntime:
    """Append-only trace helper for the navigation harness."""

    def __init__(self, emit: Callable[..., None], strict_receipts: bool = True,
                 require_fresh_observation: bool = True,
                 registry: Optional[ToolRegistry] = None,
                 trace_phases: bool = False):
        # strict_receipts is accepted and recorded but not yet enforced: nothing
        # currently refuses a world mutation that failed to produce a receipt.
        # Kept so EVAL.HARNESS_STRICT_RECEIPTS has somewhere to land, and so the
        # gap is visible here rather than looking like the flag works.
        self.emit = emit
        self.strict_receipts = bool(strict_receipts)
        self.require_fresh_observation = bool(require_fresh_observation)
        self.registry = registry if registry is not None else default_registry()
        self.trace_phases = bool(trace_phases)
        self._call_seq = 0
        self._receipt_seq = 0
        self._obs_seq = 0
        # The two evidence namespaces, per episode. Membership is what makes
        # "this claim points at something real" an O(1) check rather than a
        # guess -- see observation() and has_evidence(). The value is the step
        # the id was issued at: membership alone cannot tell whether a TODO
        # completion is backed by something the agent saw *after* taking on the
        # obligation, or by the opening panorama it had already seen when it
        # wrote the plan. See evidence_step() and TodoMemory._check_evidence.
        self._issued_obs: Dict[str, int] = {}
        self._issued_receipts: Dict[str, int] = {}
        self.phase: Optional[Phase] = None
        self.state = EpisodeState()

    def start_episode(self, episode_id: Any, step: int = 0, **fields: Any) -> None:
        self._call_seq = 0
        self._receipt_seq = 0
        self._obs_seq = 0
        self._issued_obs.clear()
        self._issued_receipts.clear()
        self.phase = None
        self.state = EpisodeState(episode_id=str(episode_id))
        self._emit("HARNESS_EPISODE_START", step, **fields)

    # State machine.
    def set_phase(self, phase, step: int, reason: Optional[str] = None) -> None:
        """Record a state transition; emit phase events only when trace_phases is enabled."""
        new_phase = phase if isinstance(phase, Phase) else Phase(str(phase))
        if new_phase == self.phase:
            return
        old_phase = self.phase
        self.phase = new_phase
        if self.trace_phases:
            self._emit("HARNESS_PHASE", step, phase=new_phase.value,
                       previous=old_phase.value if old_phase else None, reason=reason)

    # ------------------------------------------------------------------
    # Tool call / result / receipt
    # ------------------------------------------------------------------
    def call(self, name: str, step: int, args: Optional[Dict[str, Any]] = None,
             read_only: bool = False, reason: Optional[str] = None) -> ToolCall:
        self._call_seq += 1
        call = ToolCall(
            call_id=f"tc-{self._call_seq:06d}",
            name=name,
            step=int(step),
            args=args or {},
            read_only=bool(read_only),
            reason=reason,
        )
        # Report-only by design: raising here would abort a rollout over a
        # bookkeeping mistake, and refusing the call would change navigation.
        # The point is that the mismatch stops being invisible.
        for kind, message in self.registry.validate(name, call.args, read_only=call.read_only):
            self.violation(step, kind, message, tool_name=name)
        self.state.tool_calls += 1
        self.state.open_calls[call.call_id] = call
        self._emit("HARNESS_TOOL_CALL", step, **call.to_dict())
        return call

    def result(self, call: ToolCall, step: int, ok: bool = True,
               result: Optional[Dict[str, Any]] = None,
               error: Optional[str] = None) -> ToolResult:
        self.state.tool_results += 1
        self.state.open_calls.pop(call.call_id, None)
        tool_result = ToolResult(
            call_id=call.call_id,
            name=call.name,
            step=int(step),
            ok=bool(ok),
            result=result or {},
            error=error,
        )
        self._emit("HARNESS_TOOL_RESULT", step, **tool_result.to_dict())
        return tool_result

    def receipt(self, call: ToolCall, step_before: int, step_after: int,
                mutation: str, metadata: Optional[Dict[str, Any]] = None) -> Receipt:
        self._receipt_seq += 1
        rec = Receipt(
            receipt_id=f"rc-{self._receipt_seq:06d}",
            call_id=call.call_id,
            tool_name=call.name,
            step_before=int(step_before),
            step_after=int(step_after),
            mutation=mutation,
            metadata=metadata or {},
        )
        self.state.receipts += 1
        self._issued_receipts[rec.receipt_id] = int(step_after)
        self._emit("HARNESS_RECEIPT", step_after, **rec.to_dict())
        if self.require_fresh_observation and not self._is_terminal_tool(call.name):
            self.set_fresh_observation_required(step_after, reason=f"{call.name}:{mutation}")
        return rec

    # Observation evidence.
    def observation(self, step: int, kind: str, **fields: Any) -> str:
        """Issue an ``ob-`` id for one thing the agent saw, and return it.

        ``call_id`` and ``receipt_id`` cover "I asked" and "I acted"; this
        covers "I saw", which is what a TODO completion or a reflection record
        actually needs to point at. Without it ``evidence_ids`` has nothing
        addressable to hold and degenerates into prose.

        The image itself never enters the trace -- only the id, the kind and
        whatever small scalars the caller passes.
        """
        self._obs_seq += 1
        obs_id = f"ob-{self._obs_seq:06d}"
        self._issued_obs[obs_id] = int(step)
        self._emit("HARNESS_OBSERVATION", step, obs_id=obs_id, kind=str(kind), **fields)
        return obs_id

    def observations(self, step: int, kind: str, count: int, **fields: Any) -> list:
        """Issue `count` ids for one multi-frame observation (e.g. a panorama)."""
        return [self.observation(step, kind, frame_index=i, **fields)
                for i in range(max(0, int(count)))]

    def has_evidence(self, evidence_id: Any) -> bool:
        """Return whether this episode has issued the observation or receipt ID."""
        if not isinstance(evidence_id, str):
            return False
        return evidence_id in self._issued_obs or evidence_id in self._issued_receipts

    def evidence_step(self, evidence_id: Any) -> Optional[int]:
        """Return the issue step of an episode evidence ID, or None if it is unknown."""
        if not isinstance(evidence_id, str):
            return None
        if evidence_id in self._issued_obs:
            return self._issued_obs[evidence_id]
        return self._issued_receipts.get(evidence_id)

    def _is_terminal_tool(self, name: str) -> bool:
        """Terminal tools are exempt from the fresh-observation obligation.
        
        If registry metadata is unavailable, the literal tool name "stop" is terminal.
        """
        spec = self.registry.get(name)
        if spec is None:
            return name == "stop"
        return spec.side_effect == SideEffect.TERMINAL

    # Obligations.
    def set_fresh_observation_required(self, step: int, reason: str) -> None:
        self.state.fresh_observation_required = True
        self.state.fresh_observation_reason = reason
        self._emit("HARNESS_OBLIGATION_SET", step, obligation="fresh_observation", reason=reason)

    def clear_fresh_observation(self, step: int, source: str) -> None:
        if not self.state.fresh_observation_required:
            return
        old_reason = self.state.fresh_observation_reason
        self.state.fresh_observation_required = False
        self.state.fresh_observation_reason = None
        self._emit(
            "HARNESS_OBLIGATION_CLEARED", step,
            obligation="fresh_observation", source=source, reason=old_reason,
        )

    def violation(self, step: int, kind: str, message: str, **fields: Any) -> None:
        self.state.violations += 1
        self._emit("HARNESS_VIOLATION", step, kind=kind, message=message, **fields)

    def note(self, step: int, kind: str, message: str, **fields: Any) -> None:
        """Record a diagnostic note without changing episode violation counters."""
        self._emit("HARNESS_NOTE", step, kind=kind, message=message, **fields)

    def terminal(self, step: int, reason: str, **fields: Any) -> bool:
        if self.state.terminal_emitted:
            self.violation(step, "duplicate_terminal", "terminal emitted more than once", reason=reason)
            return False
        if self.state.open_calls:
            self.violation(
                step, "open_tool_calls", "terminal emitted with unpaired tool calls",
                open_call_ids=sorted(self.state.open_calls),
            )
        self.state.terminal_emitted = True
        self.phase = Phase.TERMINAL
        self._emit("HARNESS_TERMINAL", step, reason=reason, summary=self.summary(), **fields)
        return True

    def summary(self) -> Dict[str, Any]:
        return {
            "episode_id": self.state.episode_id,
            "tool_calls": self.state.tool_calls,
            "tool_results": self.state.tool_results,
            "receipts": self.state.receipts,
            "observations": len(self._issued_obs),
            "violations": self.state.violations,
            "terminal_emitted": self.state.terminal_emitted,
            "fresh_observation_required": self.state.fresh_observation_required,
            "open_tool_calls": len(self.state.open_calls),
        }

    def _emit(self, event_type: str, event_step: int, **fields: Any) -> None:
        try:
            self.emit(event_type, int(event_step), **fields)
        except Exception:
            # Diagnostics must never break evaluation.
            pass
