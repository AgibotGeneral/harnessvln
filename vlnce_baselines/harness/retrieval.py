"""Local semantic retrieval and reflection over the world graph.

Rank places against the current subgoal using IDF-weighted token cosine
similarity, recency, salience and bounded failure penalties. Queries and room
summaries share the planner's vocabulary, so retrieval needs no model request.
The selected places and prior failures are rendered into the planner prompt.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

SCHEMA_RECALL = "harness.recall.v1"

# Assembly shape, "top-3 seeds, expand one hop, keep at most 5".
DEFAULT_SEEDS = 3
DEFAULT_MAX_PLACES = 5

# Scoring weights:
#   score = 1.0 * relevance + 0.3 * recency + 0.3 * salience - repeat_failure_penalty
DEFAULT_RECENCY_WEIGHT = 0.3
DEFAULT_SALIENCE_WEIGHT = 0.3
DEFAULT_FAILURE_PENALTY_PER = 0.5

# A place with this many objects-plus-exits is as salient as it gets. Six is a
# corridor junction with a few things in view; the point of saturating is that a
# cluttered room should not outrank a junction just by containing more sofas.
SALIENCE_SATURATION = 6.0

# The penalty ceiling that keeps a reflection advisory. One full unit of
# relevance is the most a repeatedly-failed branch can be docked.
PENALTY_CAP = 1.0

# Keep retrieved context within a compact rendering budget.
MAX_OBJECTS_PER_PLACE = 4
MAX_OBS_IDS_PER_PLACE = 2
MAX_LABEL_CHARS = 48

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately short: this is a stop list for room/object prose, not for English.
# Dropping too much would cost real signal ("go to the room with the plant"
# keeps room and plant either way).
_STOPWORDS = frozenset("""
a an the this that these those there here it its is are was were be been being
of in on at to from by for with without into onto near next and or but not no
you your i we my me it's find go goes going walk walked move moves moving turn
turns turning look looks looking see seen seeing head heading continue toward
towards through past then first second third step steps area room place side
""".split())


def tokenize(text: Any) -> List[str]:
    """Lowercase alphanumeric tokens, stop words dropped, order preserved."""
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(str(text).lower()) if t not in _STOPWORDS]


class _Corpus:
    """IDF over this episode's place texts, plus each place's token set.

    Built per recall call. The corpus is the episode, not the dataset: a word is
    informative here if it distinguishes one room the agent has seen from the
    others it has seen, which is exactly the discrimination the readout needs.
    """

    def __init__(self, texts: Dict[str, str]):
        self.tokens: Dict[str, set] = {pid: set(tokenize(t)) for pid, t in texts.items()}
        n_docs = sum(1 for toks in self.tokens.values() if toks)
        self.n_docs = n_docs
        df: Dict[str, int] = {}
        for toks in self.tokens.values():
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self._df = df
        self.has_text = n_docs > 0

    def idf(self, token: str) -> float:
        # Smoothed, and never zero: a token seen in every place still carries a
        # little weight, so a one-place episode does not score 0 against itself.
        return math.log((1.0 + self.n_docs) / (1.0 + self._df.get(token, 0))) + 1.0

    def norm(self, tokens) -> float:
        return math.sqrt(sum(self.idf(t) ** 2 for t in tokens))


def relevance(query_tokens: Sequence[str], place_tokens, corpus: _Corpus) -> float:
    """IDF-weighted cosine between the query and one place, in [0, 1]."""
    q = set(query_tokens)
    p = set(place_tokens)
    if not q or not p:
        return 0.0
    shared = q & p
    if not shared:
        return 0.0
    num = sum(corpus.idf(t) ** 2 for t in shared)
    den = corpus.norm(q) * corpus.norm(p)
    if den <= 0.0:
        return 0.0
    return max(0.0, min(1.0, num / den))


def recency(place, turn: int) -> float:
    """1.0 for the place seen this turn, decaying linearly to ~0 for the first."""
    if turn <= 0:
        return 1.0
    return max(0.0, min(1.0, float(place.last_seen_turn) / float(turn)))


def salience(place) -> float:
    """Objects plus known exits, saturating. A junction beats a dead end."""
    n = len(place.object_ids) + len(place.adjacent)
    return max(0.0, min(1.0, n / SALIENCE_SATURATION))


def repeat_failure_penalty(graph, place_id: str, subgoal: str,
                           per_failure: float = DEFAULT_FAILURE_PENALTY_PER) -> float:
    """Cost of having failed *at this place, for this subgoal*, capped.

    The only term that directly presses on SPL: it is what stops the agent
    re-proposing a branch it has already exhausted. Capped at
    :data:`PENALTY_CAP` and read through ``failures_at(applicable_only=True)``,
    so an expired reflection stops costing anything.
    """
    n = graph.failure_count(place_id, subgoal=subgoal or None, applicable_only=True)
    return min(PENALTY_CAP, max(0.0, per_failure) * n)


@dataclass
class RecalledPlace:
    """One place in the readout, with the arithmetic that put it there."""

    place_id: str
    score: float
    relevance: float
    recency: float
    salience: float
    penalty: float
    seed: bool
    distance_m: Optional[float] = None
    waypoint_id: Optional[int] = None
    failures: List[Any] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            'place_id': self.place_id,
            'score': round(float(self.score), 4),
            'relevance': round(float(self.relevance), 4),
            'penalty': round(float(self.penalty), 4),
            'seed': bool(self.seed),
            'waypoint_id': self.waypoint_id,
            'n_failures': len(self.failures),
        }


@dataclass
class Recall:
    """The result of one ``recall_place`` call: what to show, and the accounting."""

    query: str
    places: List[RecalledPlace] = field(default_factory=list)
    current_place_id: Optional[str] = None
    current_failures: List[Any] = field(default_factory=list)
    n_places_total: int = 0
    # True when relevance could not be computed at all (no place has any text
    # yet, or the query has no usable tokens) and the ranking fell back to
    # recency + salience. retrieval degrading must not kill the episode,
    # but it has to be recorded -- "retrieval quietly not working" would
    # otherwise look exactly like "retrieval does not help".
    degraded: bool = False
    degraded_reason: Optional[str] = None
    text: str = ""

    def summary(self) -> Dict[str, Any]:
        """Compact, JSON-friendly accounting for the trace."""
        return {
            'schema': SCHEMA_RECALL,
            'query': self.query[:120],
            'n_recalled': len(self.places),
            'n_places_total': int(self.n_places_total),
            'n_chars': len(self.text),
            'degraded': bool(self.degraded),
            'degraded_reason': self.degraded_reason,
            'current_place_id': self.current_place_id,
            'places': [p.summary() for p in self.places],
        }


def recall(graph, query: str, current_xz=None,
           seeds: int = DEFAULT_SEEDS,
           max_places: int = DEFAULT_MAX_PLACES,
           recency_weight: float = DEFAULT_RECENCY_WEIGHT,
           salience_weight: float = DEFAULT_SALIENCE_WEIGHT,
           failure_penalty_per: float = DEFAULT_FAILURE_PENALTY_PER,
           max_backtrack_distance_m: Optional[float] = None,
           waypoint_ids_available: Optional[Sequence[int]] = None) -> Recall:
    """Rank this episode's places against `query` and render a planner readout.

    The current place is excluded from the recalled list -- the planner is
    looking at it, so spending tokens describing it back is waste -- but its
    failures are carried separately, because "you already tried to stop here and
    were wrong" is the single most useful thing to say about where you stand.
    """
    query = str(query or '')
    out = Recall(query=query, current_place_id=graph.current_place_id,
                 n_places_total=len(graph.places))

    candidates = [pid for pid in graph.places
                  if pid != graph.current_place_id and _has_content(graph, pid)]
    if graph.current_place_id:
        out.current_failures = graph.failures_at(graph.current_place_id, subgoal=query or None)
    if not candidates:
        out.text = _render(out, graph)
        return out

    texts = {pid: graph.place_text(pid) for pid in candidates}
    corpus = _Corpus(texts)
    q_tokens = tokenize(query)
    if not corpus.has_text:
        out.degraded, out.degraded_reason = True, 'no_place_text'
    elif not q_tokens:
        out.degraded, out.degraded_reason = True, 'empty_query'

    scored: Dict[str, RecalledPlace] = {}
    for pid in candidates:
        place = graph.places[pid]
        rel = 0.0 if out.degraded else relevance(q_tokens, corpus.tokens[pid], corpus)
        rec = recency(place, graph.turn)
        sal = salience(place)
        pen = repeat_failure_penalty(graph, pid, query, failure_penalty_per)
        scored[pid] = RecalledPlace(
            place_id=pid,
            score=rel + recency_weight * rec + salience_weight * sal - pen,
            relevance=rel, recency=rec, salience=sal, penalty=pen, seed=False,
        )

    ranked = sorted(scored.values(), key=lambda r: (-r.score, r.place_id))
    chosen: List[RecalledPlace] = []
    for entry in ranked[:max(0, int(seeds))]:
        entry.seed = True
        chosen.append(entry)

    # One hop out from each seed, plus the current place's neighbours: the
    # topology is the reason a graph beats a list here. A seed's neighbour is
    # somewhere the agent can actually get to from a place it was just reminded
    # of, which is what makes the readout actionable.
    expand: List[str] = []
    for entry in chosen:
        expand.extend(graph.neighbours(entry.place_id))
    if graph.current_place_id:
        expand.extend(graph.neighbours(graph.current_place_id))
    seen = {e.place_id for e in chosen}
    extras = []
    for pid in expand:
        if pid in seen or pid not in scored:
            continue
        seen.add(pid)
        extras.append(scored[pid])
    chosen.extend(sorted(extras, key=lambda r: (-r.score, r.place_id)))
    chosen = chosen[:max(1, int(max_places))]
    chosen.sort(key=lambda r: (-r.score, r.place_id))

    allowed = set(int(w) for w in (waypoint_ids_available or [])) \
        if waypoint_ids_available is not None else None
    for entry in chosen:
        place = graph.places[entry.place_id]
        entry.distance_m = graph.distance_from(entry.place_id, current_xz)
        entry.failures = graph.failures_at(entry.place_id, subgoal=query or None)
        entry.waypoint_id = _backtrackable_waypoint(
            place, entry.distance_m, max_backtrack_distance_m, allowed)

    out.places = chosen
    out.text = _render(out, graph)
    return out


def _has_content(graph, place_id: str) -> bool:
    """Whether a place has anything worth recalling.

    A node the agent passed through without grounding an object, without a
    summary and without failing there is a coordinate, not a memory. Rendering
    it back would spend tokens on ``place-4 [3.0 m away, visited 1x]``, which
    tells the planner nothing it can act on or cite. Filtered here rather than
    in the renderer so the score list, the trace and the prompt agree on what
    was recalled.
    """
    place = graph.places.get(place_id)
    if place is None:
        return False
    return bool(place.semantic_summary or place.object_ids
                or place.failure_ids or place.observation_ids)


def _backtrackable_waypoint(place, distance_m: Optional[float],
                            max_distance_m: Optional[float],
                            allowed: Optional[set]) -> Optional[int]:
    """The waypoint id the planner may name for this place, if any.

    Rendering a waypoint id the action list does not offer would invite an
    illegal ``backtrack to N``, so the same distance rule the action list uses
    gates it here. This is why ``MAX_BACKTRACK_DISTANCE_M`` is config rather
    than a literal in two places.
    """
    if not place.waypoint_ids:
        return None
    wp = min(int(w) for w in place.waypoint_ids)
    if allowed is not None and wp not in allowed:
        return None
    if max_distance_m is not None:
        if distance_m is None or distance_m > float(max_distance_m):
            return None
    return wp


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _clip(text: Any, limit: int = MAX_LABEL_CHARS) -> str:
    s = " ".join(str(text or '').split())
    return s if len(s) <= limit else s[:limit - 1] + '…'


def _failure_phrase(failures) -> str:
    """`NO_PROGRESS x2 (target timeout), STOP_REJECTED x1` -- types then detail."""
    order: List[str] = []
    counts: Dict[str, int] = {}
    details: Dict[str, str] = {}
    for rec in failures:
        ftype = str(rec.failure_type)
        if ftype not in counts:
            order.append(ftype)
            counts[ftype] = 0
        counts[ftype] += 1
        if rec.detail and ftype not in details:
            details[ftype] = _clip(rec.detail, 32)
    parts = []
    for ftype in order:
        part = f"{ftype} x{counts[ftype]}"
        if ftype in details:
            part += f" ({details[ftype]})"
        parts.append(part)
    return ", ".join(parts)


def _render(out: Recall, graph) -> str:
    """The prompt block. Empty string when there is nothing worth saying.

    Everything rendered here is either something the planner can act on (a
    waypoint id it may backtrack to, a branch that already failed for this
    subgoal) or something it can cite (an ``ob-`` id for ``evidence_ids``).
    Descriptive filler is left out on purpose: the readout has to be worth its
    tokens against the ~11k a planner call already costs.
    """
    lines: List[str] = []
    any_failure = False
    for entry in out.places:
        place = graph.places[entry.place_id]
        head = [f"visited {place.visit_count}x", f"last seen step {place.last_seen_step}"]
        if entry.distance_m is not None:
            head.insert(0, f"{entry.distance_m:.1f} m away")
        if entry.waypoint_id is not None:
            head.insert(0, f"backtrack id {entry.waypoint_id}")
        lines.append(f"- {entry.place_id} [{', '.join(head)}]")
        if place.semantic_summary:
            lines.append(f"    {_clip(place.semantic_summary, 96)}")
        labels = []
        for obj_id in place.object_ids[:MAX_OBJECTS_PER_PLACE]:
            obj = graph.objects.get(obj_id)
            if obj is not None:
                labels.append(_clip(obj.category))
        if labels:
            lines.append(f"    seen here: {', '.join(labels)}")
        obs = place.observation_ids[:MAX_OBS_IDS_PER_PLACE]
        if obs:
            lines.append(f"    views you can cite: {', '.join(obs)}")
        if entry.failures:
            any_failure = True
            lines.append(f"    ALREADY FAILED here for this sub-goal: "
                         f"{_failure_phrase(entry.failures)}")
    if out.current_failures:
        any_failure = True
        lines.append(f"- where you stand now: ALREADY FAILED here for this sub-goal: "
                     f"{_failure_phrase(out.current_failures)}")
    if not lines:
        return ""
    header = (f"Episode memory (recalled {len(out.places)} of {out.n_places_total} "
              f"visited places, most relevant first)")
    if out.degraded:
        # Say so in the prompt as well as the trace: an ordering that is not
        # relevance-based should not be presented to the planner as if it were.
        header += " -- ranked by recency only, no semantic match available"
    text = f"{header}:\n" + "\n".join(lines)
    if any_failure:
        # Only spent when there is a failure to qualify. advisory, not a
        # ban -- so the sentence has to leave the door open explicitly, or a
        # model reading "FAILED" will treat the place as off-limits and the
        # wrong-STOP failures will just turn into timeouts.
        text += ("\nA place marked ALREADY FAILED was tried for this same sub-goal and did "
                 "not work out; prefer somewhere else unless you can say what would be "
                 "different this time. This is a reminder, not a restriction.")
    return text
