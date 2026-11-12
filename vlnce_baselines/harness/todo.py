"""TODO memory with stable identifiers and optional evidence validation.

Each item has a persistent ``todo_id``. Removal creates a tombstone, and
index-based updates resolve against a snapshot taken before the batch.
Statuses are ``PENDING``, ``ACTIVE``, ``COMPLETED`` and ``BLOCKED``.

Strict evidence mode requires an observation or receipt identifier accepted
by the evidence resolver. Non-strict mode also accepts a written result and
reports missing evidence. Freshness checks emit diagnostic notes without
rejecting updates. Configure strict checking with ``EVAL.TODO_EVIDENCE_STRICT``.

This module has no simulator or tensor-library dependencies.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

SCHEMA_TODO = "harness.todo.v1"

PENDING = "PENDING"
ACTIVE = "ACTIVE"
COMPLETED = "COMPLETED"
BLOCKED = "BLOCKED"

TODO_STATUSES: Tuple[str, ...] = (PENDING, ACTIVE, COMPLETED, BLOCKED)

# Accepted status spellings; unrecognized values are ignored.
_STATUS_ALIASES: Dict[str, str] = {
    'pending': PENDING,
    'todo': PENDING,
    'open': PENDING,
    'active': ACTIVE,
    'in_progress': ACTIVE,
    'in progress': ACTIVE,
    'doing': ACTIVE,
    'completed': COMPLETED,
    'complete': COMPLETED,
    'done': COMPLETED,
    'blocked': BLOCKED,
    'unreachable': BLOCKED,
}


def normalize_status(value: Any) -> Optional[str]:
    """Map a model-supplied status onto the enum, or None if it is not one."""
    if not isinstance(value, str):
        return None
    return _STATUS_ALIASES.get(value.strip().lower())


def is_evidence_id(value: Any) -> bool:
    """Whether `value` has the shape of a harness evidence id (``ob-`` / ``rc-``).

    Shape only. Whether the id was actually issued for this episode is the
    resolver's job -- this predicate is what filters prose out of
    ``evidence_ids`` before the resolver is bothered with it.
    """
    return isinstance(value, str) and (value.startswith('ob-') or value.startswith('rc-'))


class TodoMemory:
    """Episode TODO items with stable IDs, statuses and evidence checks.
    
    ``on_event`` reports rejected or suspicious updates. ``on_note`` records
    freshness diagnostics separately, without incrementing violation counts.
    """

    def __init__(self,
                 evidence_resolver: Optional[Callable[[str], bool]] = None,
                 strict_evidence: bool = False,
                 on_event: Optional[Callable[..., None]] = None,
                 logger: Any = None,
                 evidence_step_resolver: Optional[Callable[[str], Optional[int]]] = None,
                 on_note: Optional[Callable[..., None]] = None):
        self.evidence_resolver = evidence_resolver
        self.evidence_step_resolver = evidence_step_resolver
        self.strict_evidence = bool(strict_evidence)
        self.on_event = on_event
        self.on_note = on_note
        self.logger = logger
        self.reset()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the episode's plan. Leaves the resolver/logger wiring in place:
        those are per-agent, the plan is per-episode.
        """
        self._items: List[Dict[str, Any]] = []
        self._seq = 0
        self._initialised = False
        self.legacy_text: Optional[str] = None
        self.turn = 0
        self.step = 0
        self.last_report: List[Dict[str, Any]] = []

    @property
    def initialised(self) -> bool:
        """Return whether a plan has been loaded."""
        return self._initialised

    def load(self, parsed_items: List[Dict[str, Any]], turn: int = 0,
             step: Optional[int] = None) -> None:
        """Install a freshly generated plan, assigning ids in order.

        `step` is the sim step the plan was written at, recorded per item as
        ``created_step``. It matters because the plan is generated *from* the
        opening panorama: without it every item looks as if it were created at
        step 0, and closing one with panorama evidence would look contemporaneous
        instead of retrospective. See :meth:`_check_freshness`.
        """
        self.reset()
        self.turn = int(turn)
        if step is not None:
            self.step = int(step)
        for raw in parsed_items or []:
            if not isinstance(raw, dict) or not str(raw.get('content', '')).strip():
                continue
            self._items.append(self._new_item(
                content=str(raw['content']),
                status=normalize_status(raw.get('status')) or PENDING,
                result=str(raw.get('result', '')),
            ))
        self._initialised = True

    def load_legacy_text(self, text: str) -> None:
        """Accept the pre-structured markdown TODO (rendered through verbatim)."""
        self.legacy_text = str(text)
        self._initialised = True

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def visible(self) -> List[Dict[str, Any]]:
        """Items the planner is shown: everything not tombstoned."""
        return [t for t in self._items if not t.get('removed')]

    def as_list(self) -> Optional[Any]:
        """Return items or a Markdown plan for logging, or None before initialization."""
        if not self._initialised:
            return None
        if self.legacy_text is not None:
            return self.legacy_text
        return [dict(t) for t in self.visible()]

    def get(self, todo_id: str) -> Optional[Dict[str, Any]]:
        for t in self._items:
            if t['todo_id'] == todo_id:
                return t
        return None

    def active(self) -> Optional[Dict[str, Any]]:
        for t in self.visible():
            if t['status'] == ACTIVE:
                return t
        return None

    def render(self) -> str:
        """The prompt block. Addresses items by ``todo_id``, never by position."""
        if self.legacy_text is not None:
            return self.legacy_text
        items = self.visible()
        if not items:
            return "(empty)"
        lines = []
        for t in items:
            line = f"[{t['todo_id']}] ({t['status']})  {t['content']}"
            result = t.get('result', '')
            if t['status'] in (COMPLETED, BLOCKED) and result:
                line += f"  => {result}"
            ev = [e for e in t.get('evidence_ids', []) if e]
            if ev:
                line += f"  [evidence: {', '.join(ev)}]"
            lines.append(line)
        return "\n".join(lines)

    def snapshot(self) -> Dict[str, Any]:
        """Compact, JSON-friendly state for the trace."""
        counts = {s: 0 for s in TODO_STATUSES}
        for t in self.visible():
            counts[t['status']] = counts.get(t['status'], 0) + 1
        return {
            'schema': SCHEMA_TODO,
            'total': len(self._items),
            'visible': len(self.visible()),
            'removed': sum(1 for t in self._items if t.get('removed')),
            'by_status': counts,
            'active_id': (self.active() or {}).get('todo_id'),
        }

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def apply(self, updates: Any, turn: Optional[int] = None,
              step: Optional[int] = None) -> List[Dict[str, Any]]:
        """Apply one batch of planner updates. Returns a per-op report.

        The report is what makes rejections visible: each entry carries the op,
        the resolved ``todo_id`` and either ``ok`` or a ``rejected`` reason. It
        is also what the caller turns into planner feedback, so a rejection is
        something the model is told about rather than something that silently
        did not happen.
        """
        self.last_report = []
        if turn is not None:
            self.turn = int(turn)
        if step is not None:
            self.step = int(step)
        if not updates or not isinstance(updates, list):
            return self.last_report
        if self.legacy_text is not None:
            self._report('update', None, rejected='legacy_text_todo')
            return self.last_report

        # Resolve every `index` against this, not against the list as it is
        # being mutated: the model wrote its indices against the prompt.
        pre_batch = list(self.visible())

        for u in updates:
            if not isinstance(u, dict):
                self._report('?', None, rejected='not_an_object')
                continue
            op = str(u.get('op', 'update') or 'update').lower()
            try:
                if op in ('add', 'insert'):
                    self._op_add(u)
                elif op == 'remove':
                    self._op_remove(u, pre_batch)
                elif op == 'rewrite':
                    self._op_rewrite(u, pre_batch)
                elif op == 'update':
                    self._op_update(u, pre_batch)
                else:
                    self._report(op, None, rejected='unknown_op')
            except Exception as e:  # never let bookkeeping abort a rollout
                self._log('warning', f"todo update skipped {u}: {e}")
                self._report(op, None, rejected=f'exception:{e}')
        return self.last_report

    # -- individual ops -------------------------------------------------
    def _op_add(self, u: Dict[str, Any]) -> None:
        content = str(u.get('content', '')).strip()
        if not content:
            self._report('add', None, rejected='empty_content')
            return
        item = self._new_item(
            content=content,
            status=normalize_status(u.get('status')) or PENDING,
            result=str(u.get('result', '')),
        )
        # ACTIVE-on-create would bypass the single-active rule; start PENDING and
        # let a normal update promote it.
        if item['status'] == ACTIVE and self.active() is not None:
            item['status'] = PENDING
            self._report('add', item['todo_id'], rejected='multiple_active')
        # Position: `after` is the id-addressed form and wins; `index` is the
        # legacy positional one, kept so a plan written against the old prompt
        # still lands where it meant to.
        after = u.get('after')
        pos = u.get('index')
        note = None
        if isinstance(after, str) and after.strip():
            anchor = self.get(after.strip())
            if anchor is None or anchor.get('removed'):
                # Appending is the safe fallback: the item is still wanted, only
                # its position is unknown, so this is a warning and not a refusal.
                note = 'unknown_after_id'
                self._items.append(item)
            else:
                self._items.insert(self._items.index(anchor) + 1, item)
        elif isinstance(pos, int) and not isinstance(pos, bool):
            self._insert_before_visible(item, pos)
        else:
            self._items.append(item)
        self._report('add', item['todo_id'], ok=True,
                     **({'warning': note} if note else {}))

    def _op_remove(self, u: Dict[str, Any], pre_batch: List[Dict]) -> None:
        item = self._resolve(u, pre_batch, 'remove')
        if item is None:
            return
        # Tombstone, never pop: the id must stay dead rather than becoming
        # somebody else's.
        item['removed'] = True
        item['updated_turn'] = self.turn
        self._report('remove', item['todo_id'], ok=True)

    def _op_rewrite(self, u: Dict[str, Any], pre_batch: List[Dict]) -> None:
        item = self._resolve(u, pre_batch, 'rewrite')
        if item is None:
            return
        if item['status'] == COMPLETED:
            self._report('rewrite', item['todo_id'], rejected='already_completed')
            return
        content = str(u.get('content', '')).strip()
        if content:
            item['content'] = content
        if 'result' in u:
            item['result'] = str(u.get('result') or '')
        item['updated_turn'] = self.turn
        self._report('rewrite', item['todo_id'], ok=True)

    def _op_update(self, u: Dict[str, Any], pre_batch: List[Dict]) -> None:
        item = self._resolve(u, pre_batch, 'update')
        if item is None:
            return
        todo_id = item['todo_id']
        new_result = u.get('result')
        if new_result is None:
            # The prompt asks for `reason` on a BLOCKED update and `result`
            # everywhere else. Both are "the sentence backing this status", so
            # both land in the same field -- otherwise every blocked update the
            # prompt asks for would be refused as blocked_without_reason.
            new_result = u.get('reason')
        raw_status = u.get('status')
        new_status = normalize_status(raw_status)
        if raw_status is not None and new_status is None:
            self._report('update', todo_id, rejected='unknown_status')

        # `result` lands first: the evidence check for COMPLETED reads it.
        if new_result is not None:
            item['result'] = str(new_result)
        ev = self._collect_evidence(u)

        if new_status is not None and new_status != item['status']:
            if not self._transition(item, new_status, ev, u):
                item['updated_turn'] = self.turn
                return
        elif ev:
            self._merge_evidence(item, ev)

        item['updated_turn'] = self.turn
        self._report('update', todo_id, ok=True, status=item['status'])

    # -- transitions ----------------------------------------------------
    def _transition(self, item: Dict[str, Any], new_status: str,
                    evidence: List[str], u: Dict[str, Any]) -> bool:
        """Attempt a status change. Returns False if it was refused."""
        todo_id = item['todo_id']
        old = item['status']

        if new_status == ACTIVE:
            other = self.active()
            if other is not None and other['todo_id'] != todo_id:
                # Refusing is safe because PENDING -> COMPLETED is legal: an item
                # can always finish without passing through ACTIVE, so this
                # cannot deadlock the plan.
                self._report('update', todo_id, rejected='multiple_active',
                             blocking_id=other['todo_id'])
                return False

        if new_status == BLOCKED and not str(item.get('result', '')).strip():
            # Reopening a failed item requires a written reflection reason.
            self._report('update', todo_id, rejected='blocked_without_reason')
            return False

        if new_status == COMPLETED:
            ok, reason, resolved = self._check_evidence(item, evidence)
            if not ok:
                self._report('update', todo_id, rejected=reason)
                return False
            self._merge_evidence(item, resolved or evidence)
            self._check_freshness(item, resolved)
            if reason:
                # Accepted, but the claim is backed by prose rather than by an
                # id that resolves. This is the number that decides when
                # strict_evidence can be turned on.
                self._report('update', todo_id, ok=True, warned=reason)

        if old == COMPLETED and new_status != COMPLETED:
            self._report('update', todo_id, ok=True, warned='reopened')

        item['status'] = new_status
        if evidence and new_status != COMPLETED:
            self._merge_evidence(item, evidence)
        return True

    def _check_evidence(self, item: Dict[str, Any],
                        evidence: List[str]) -> Tuple[bool, Optional[str], List[str]]:
        """Validate a completion claim and return acceptance, warning and resolved IDs.
        
        Strict mode requires an evidence ID accepted by the resolver. Non-strict
        mode also accepts a written result and reports unresolved or missing evidence.
        """
        resolved = self._resolve_evidence(evidence)
        has_result = bool(str(item.get('result', '')).strip())

        if resolved:
            return True, None, resolved
        if self.strict_evidence:
            return False, ('evidence_unresolved' if evidence else 'evidence_missing'), []
        if has_result:
            return True, ('evidence_unresolved' if evidence else 'evidence_missing'), []
        # Completion requires either resolved evidence or a written result.
        return False, 'no_result_and_no_evidence', []

    def _resolve_evidence(self, evidence: List[str]) -> List[str]:
        shaped = [e for e in evidence if is_evidence_id(e)]
        if not shaped:
            return []
        if self.evidence_resolver is None:
            # No harness wired (e.g. HARNESS_ENABLED off): shape is all we can
            # check. Do not pretend the id was verified.
            return shaped
        out = []
        for e in shaped:
            try:
                if self.evidence_resolver(e):
                    out.append(e)
            except Exception:
                continue
        return out

    def _check_freshness(self, item: Dict[str, Any], resolved: List[str]) -> None:
        """Report evidence freshness without refusing completion.
        
        ``evidence_predates_item`` means every resolved evidence ID was issued before
        the item was created. ``closed_at_creation_step`` means completion occurred
        on the creation step. These are independent diagnostic signals.
        """
        created = item.get('created_step')
        if created is None:
            return
        reasons = []
        if self.step == created:
            reasons.append('closed_at_creation_step')
        steps = []
        if resolved and self.evidence_step_resolver is not None:
            for e in resolved:
                try:
                    s = self.evidence_step_resolver(e)
                except Exception:
                    continue
                if s is not None:
                    steps.append(int(s))
            if steps and max(steps) < int(created):
                reasons.append('evidence_predates_item')
        if not reasons:
            return
        self._emit_note('todo_evidence_stale',
                        f"{item['todo_id']} completed with nothing newer than the plan "
                        f"({'+'.join(reasons)})",
                        todo_id=item['todo_id'], reasons=reasons,
                        created_step=int(created), closed_step=int(self.step),
                        evidence_ids=list(resolved or []), evidence_steps=steps)

    # -- helpers --------------------------------------------------------
    def _new_item(self, content: str, status: str, result: str = '',
                  evidence_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        self._seq += 1
        return {
            'todo_id': f'todo-{self._seq}',
            'content': content,
            'status': status if status in TODO_STATUSES else PENDING,
            'result': result,
            'evidence_ids': list(evidence_ids or []),
            'created_turn': self.turn,
            'created_step': self.step,
            'updated_turn': self.turn,
            'removed': False,
        }

    def _insert_before_visible(self, item: Dict[str, Any], pos: int) -> None:
        """Insert so the item lands at visible position `pos`, tombstones skipped."""
        vis = self.visible()
        pos = max(0, min(int(pos), len(vis)))
        if pos == len(vis):
            self._items.append(item)
            return
        anchor = vis[pos]
        self._items.insert(self._items.index(anchor), item)

    def _resolve(self, u: Dict[str, Any], pre_batch: List[Dict],
                 op: str) -> Optional[Dict[str, Any]]:
        todo_id = u.get('todo_id') or u.get('id')
        if isinstance(todo_id, str) and todo_id.strip():
            item = self.get(todo_id.strip())
            if item is None:
                self._report(op, todo_id.strip(), rejected='unknown_todo_id')
                return None
            if item.get('removed'):
                self._report(op, item['todo_id'], rejected='todo_removed')
                return None
            return item

        idx = u.get('index')
        if isinstance(idx, int) and not isinstance(idx, bool):
            # Compatibility path, kept for one version. The violation is how we
            # find out whether the planner has actually switched to ids.
            self._emit('todo_index_addressing',
                       f'{op} addressed TODO by index {idx}; use todo_id',
                       op=op, index=int(idx))
            if 0 <= idx < len(pre_batch):
                item = pre_batch[idx]
                if item.get('removed'):
                    self._report(op, item['todo_id'], rejected='todo_removed')
                    return None
                return item
            self._report(op, None, rejected='index_out_of_range', index=int(idx))
            return None

        self._report(op, None, rejected='no_target')
        return None

    def _collect_evidence(self, u: Dict[str, Any]) -> List[str]:
        raw = u.get('evidence_ids', u.get('evidence'))
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(e).strip() for e in raw if str(e).strip()]

    def _merge_evidence(self, item: Dict[str, Any], evidence: List[str]) -> None:
        seen = set(item.get('evidence_ids', []))
        for e in evidence:
            if e not in seen:
                item.setdefault('evidence_ids', []).append(e)
                seen.add(e)

    def _report(self, op: str, todo_id: Optional[str], ok: bool = False,
                rejected: Optional[str] = None, **fields: Any) -> None:
        entry = {'op': op, 'todo_id': todo_id, 'ok': bool(ok)}
        if rejected:
            entry['rejected'] = rejected
        entry.update(fields)
        self.last_report.append(entry)
        if rejected:
            self._log('info', f"TODO {op} on {todo_id} rejected: {rejected}")
            self._emit('todo_update_rejected', f'{op} rejected: {rejected}',
                       op=op, todo_id=todo_id, reason=rejected)
        elif entry.get('warned'):
            # Applied, but short of the bar strict mode would set. This has to
            # reach the trace, not just `last_report`: it *is* the compliance
            # measurement that decides whether TODO_EVIDENCE_STRICT can be
            # turned on, and last_report is overwritten by the next batch.
            self._emit('todo_update_warned', f"{op} accepted with {entry['warned']}",
                       op=op, todo_id=todo_id, reason=entry['warned'])

    def rejection_feedback(self) -> str:
        """Return planner-facing feedback for rejected updates in the last batch."""
        msgs = []
        for e in self.last_report:
            if not e.get('rejected'):
                continue
            target = e.get('todo_id') or '(unresolved)'
            msgs.append(f"- {e['op']} on {target} was rejected: {e['rejected']}")
        if not msgs:
            return ""
        return ("System: some of your TODO updates were not applied:\n"
                + "\n".join(msgs)
                + "\nAddress items by their [todo-N] id, and back a COMPLETED "
                  "claim with the evidence id shown next to the view you used.")

    def _emit(self, kind: str, message: str, **fields: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(self.step, kind, message, **fields)
        except Exception:
            pass

    def _emit_note(self, kind: str, message: str, **fields: Any) -> None:
        """Emit a report-only measurement. No fallback to ``on_event`` by design
        -- see the class docstring: that path increments ``violations``.
        """
        if self.on_note is None:
            return
        try:
            self.on_note(self.step, kind, message, **fields)
        except Exception:
            pass

    def _log(self, level: str, message: str) -> None:
        if self.logger is None:
            return
        try:
            getattr(self.logger, level)(message)
        except Exception:
            pass
