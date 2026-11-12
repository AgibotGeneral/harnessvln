"""Append-only per-episode event stream.

Extracted from ``ZS_Evaluator_mp._emit_event`` / ``_close_episode_event_stream``
with the record shape, the lazy file open, the flush cadence and the
never-raise policy all preserved -- ``tools/report_rounds.py`` parses
these files and past runs must remain comparable.

The episode id and task type are read through a callable rather than captured
at construction: the evaluator mutates ``current_episode_id`` as it goes, and
the original code read it at emit time. Late binding is therefore part of the
contract, not an accident.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Optional


class EventStream:
    """One JSONL file per episode, opened on the first event."""

    def __init__(self, save_dir: str, rank: int, world_size: int,
                 context: Callable[[], Dict[str, Any]],
                 enabled: bool = True, flush_every: int = 50,
                 on_error: Optional[Callable[[str], None]] = None):
        self.save_dir = save_dir
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.context = context
        self.enabled = bool(enabled)
        self.flush_every = max(1, int(flush_every))
        self.on_error = on_error
        self._fh = None
        self._count = 0
        self._path: Optional[str] = None

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def event_count(self) -> int:
        return self._count

    def emit(self, event_type: str, step: int, **fields: Any) -> None:
        """Append one event. Never raises.

        Events are the only way to tell a wrong bbox apart from a pulled-back
        projection after the fact, so this stays on by default -- a few dozen
        lines per episode is nothing next to the VLM calls.
        """
        if not self.enabled:
            return
        try:
            ctx = self.context() or {}
            episode_id = str(ctx.get('episode_id', 'unknown'))
            if self._fh is None:
                episode_dir = os.path.join(self.save_dir, episode_id)
                os.makedirs(episode_dir, exist_ok=True)
                self._path = os.path.join(
                    episode_dir, f"events_r{self.rank}_w{self.world_size}.jsonl")
                self._fh = open(self._path, "a")
            record = {
                'ts': time.time(),
                'rank': self.rank,
                'world_size': self.world_size,
                'episode_id': episode_id,
                'task_type': ctx.get('task_type'),
                'event': event_type,
                'step': int(step),
            }
            record.update(fields)
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._count += 1
            if self._count % self.flush_every == 0:
                self._fh.flush()
        except Exception as e:  # pragma: no cover - diagnostics must not break eval
            if self.on_error is not None:
                self.on_error(f"Event stream write failed ({event_type}): {e}")

    def close(self) -> None:
        """Flush and close the current episode's file, if one is open."""
        fh = self._fh
        if fh is None:
            return
        try:
            fh.flush()
            fh.close()
        except Exception:
            pass
        finally:
            self._fh = None
