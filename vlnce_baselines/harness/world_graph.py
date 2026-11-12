"""Per-episode memory of places, objects and failed navigation attempts.

NAVIGABLE_TO edges record actual traversal; map proximity alone does not imply
connectivity. Groundings merge by category and position, and failure records
remain available to retrieval after a navigation waypoint is removed.

This module has no simulator or tensor-library dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_PLACE = "harness.place.v1"
SCHEMA_OBJECT = "harness.object.v1"
SCHEMA_WORLD_GRAPH = "harness.world_graph.v1"

# edge types.
NAVIGABLE_TO = "NAVIGABLE_TO"
CONTAINS = "CONTAINS"
OBSERVED_FROM = "OBSERVED_FROM"
FAILED_FOR_SUBGOAL = "FAILED_FOR_SUBGOAL"

EDGE_TYPES: Tuple[str, ...] = (NAVIGABLE_TO, CONTAINS, OBSERVED_FROM, FAILED_FOR_SUBGOAL)

# Failure labels describe outcomes; they do not infer object identity.
FAIL_DEAD_END = "DEAD_END"
FAIL_TARGET_MISIDENTIFIED = "TARGET_MISIDENTIFIED"
FAIL_COLLISION = "COLLISION"
FAIL_NO_PROGRESS = "NO_PROGRESS"
FAIL_STOP_REJECTED = "STOP_REJECTED"
FAIL_GROUNDING_ERROR = "GROUNDING_ERROR"

FAILURE_TYPES: Tuple[str, ...] = (
    FAIL_DEAD_END, FAIL_TARGET_MISIDENTIFIED, FAIL_COLLISION,
    FAIL_NO_PROGRESS, FAIL_STOP_REJECTED, FAIL_GROUNDING_ERROR,
)


@dataclass
class PlaceNode:
    """A physical location the agent stood at and observed from."""

    place_id: str
    world_xz: Tuple[float, float]
    map_cell: Optional[Tuple[int, int]] = None
    heading: Optional[float] = None
    first_seen_turn: int = 0
    last_seen_turn: int = 0
    first_seen_step: int = 0
    last_seen_step: int = 0
    visit_count: int = 1
    semantic_summary: str = ""
    observation_ids: List[str] = field(default_factory=list)
    object_ids: List[str] = field(default_factory=list)
    adjacent: List[str] = field(default_factory=list)
    # Waypoint indices in the legacy visited_targets list that map onto this
    # place. Keeps the graph joinable with the list while both exist.
    waypoint_ids: List[int] = field(default_factory=list)
    todo_progress: List[str] = field(default_factory=list)
    failure_ids: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            'schema': SCHEMA_PLACE,
            'place_id': self.place_id,
            'world_xz': [float(self.world_xz[0]), float(self.world_xz[1])],
            'map_cell': list(self.map_cell) if self.map_cell else None,
            'visit_count': int(self.visit_count),
            'first_seen_step': int(self.first_seen_step),
            'last_seen_step': int(self.last_seen_step),
            'semantic_summary': self.semantic_summary,
            'n_observations': len(self.observation_ids),
            'object_ids': list(self.object_ids),
            'adjacent': list(self.adjacent),
            'waypoint_ids': list(self.waypoint_ids),
            'n_failures': len(self.failure_ids),
        }


@dataclass
class ObjectNode:
    """An open-vocabulary object the agent grounded at some place."""

    object_id: str
    category: str
    description: str = ""
    place_id: Optional[str] = None
    world_xz: Optional[Tuple[float, float]] = None
    map_cell: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None
    bearing_deg: Optional[float] = None
    source: str = ""
    observation_ids: List[str] = field(default_factory=list)
    first_seen_turn: int = 0
    last_seen_turn: int = 0
    first_seen_step: int = 0
    last_seen_step: int = 0
    observation_count: int = 1
    # Other phrasings the same physical object was grounded under. grounding writes a
    # fresh sentence every turn ("the stainless steel stove in the kitchen" one
    # turn, "stove" the next), so the wording is not an identity -- position is.
    aliases: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            'schema': SCHEMA_OBJECT,
            'object_id': self.object_id,
            'category': self.category,
            'description': self.description,
            'place_id': self.place_id,
            'world_xz': ([float(self.world_xz[0]), float(self.world_xz[1])]
                         if self.world_xz else None),
            'confidence': self.confidence,
            'source': self.source,
            'observation_count': int(self.observation_count),
            'last_seen_step': int(self.last_seen_step),
            'aliases': list(self.aliases),
        }


@dataclass
class FailureRecord:
    """One failed attempt, attached to the place it was attempted from.

    The record survives waypoint removal and supplies advisory retrieval context.
    """

    failure_id: str
    place_id: str
    failure_type: str
    subgoal: str = ""
    step: int = 0
    turn: int = 0
    detail: str = ""
    object_id: Optional[str] = None
    covered_place_ids: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    # reflections are advisory and must never permanently block a place.
    # This flag is how a record stops applying; nothing consults a failure
    # without checking it.
    still_applicable: bool = True

    def summary(self) -> Dict[str, Any]:
        return {
            'failure_id': self.failure_id,
            'place_id': self.place_id,
            'failure_type': self.failure_type,
            'subgoal': self.subgoal,
            'step': int(self.step),
            'detail': self.detail,
            'object_id': self.object_id,
            'covered_place_ids': list(self.covered_place_ids),
            'evidence_ids': list(self.evidence_ids),
            'still_applicable': bool(self.still_applicable),
        }


class WorldGraph:
    """Per-episode Place/Object graph for navigation memory and retrieval.

    ``merge_radius_m`` decides when a new observation is the *same* place seen
    again. It defaults to 0.75 m to reuse the scale that already exists in the
    evaluator (``TARGET_REACHED_THRESHOLD`` = 15 cells at 5 cm/cell) rather than
    inventing a second, silently different notion of "here". Getting this wrong
    matters: without merging, ``visit_count`` never exceeds 1 and the
    repeat-failure penalty item 3 depends on is identically zero.
    """

    def __init__(self, merge_radius_m: float = 0.75,
                 object_merge_radius_m: float = 1.0,
                 same_point_radius_m: float = 0.35,
                 on_event: Optional[Any] = None):
        self.merge_radius_m = float(merge_radius_m)
        self.object_merge_radius_m = float(object_merge_radius_m)
        self.same_point_radius_m = float(same_point_radius_m)
        self.on_event = on_event
        self.reset()

    def reset(self) -> None:
        self.places: Dict[str, PlaceNode] = {}
        self.objects: Dict[str, ObjectNode] = {}
        self.failures: Dict[str, FailureRecord] = {}
        self.edges: List[Dict[str, Any]] = []
        self._place_seq = 0
        self._object_seq = 0
        self._failure_seq = 0
        self.current_place_id: Optional[str] = None
        self.turn = 0
        self._edge_index: set = set()

    # ------------------------------------------------------------------
    # places
    # ------------------------------------------------------------------
    def observe_place(self, world_xz, step: int, map_cell=None, heading=None,
                      observation_ids: Optional[List[str]] = None,
                      waypoint_id: Optional[int] = None,
                      semantic_summary: str = "",
                      link_from_current: bool = True) -> PlaceNode:
        """Record standing at `world_xz` and looking around.

        Merges into an existing place when within ``merge_radius_m``, otherwise
        creates one. Adds a ``NAVIGABLE_TO`` edge from the previous place when
        the agent actually moved between the two.
        """
        self.turn += 1
        existing = self._nearest_place(world_xz, self.merge_radius_m)
        if existing is None:
            self._place_seq += 1
            place = PlaceNode(
                place_id=f"place-{self._place_seq}",
                world_xz=(float(world_xz[0]), float(world_xz[1])),
                map_cell=tuple(int(c) for c in map_cell) if map_cell is not None else None,
                heading=float(heading) if heading is not None else None,
                first_seen_turn=self.turn, last_seen_turn=self.turn,
                first_seen_step=int(step), last_seen_step=int(step),
                semantic_summary=semantic_summary,
            )
            self.places[place.place_id] = place
            created = True
        else:
            place = existing
            place.visit_count += 1
            place.last_seen_turn = self.turn
            place.last_seen_step = int(step)
            if heading is not None:
                place.heading = float(heading)
            if semantic_summary:
                place.semantic_summary = semantic_summary
            created = False

        for obs_id in observation_ids or []:
            if obs_id not in place.observation_ids:
                place.observation_ids.append(obs_id)
                self._add_edge(OBSERVED_FROM, place.place_id, obs_id, step=step)
        if waypoint_id is not None and int(waypoint_id) not in place.waypoint_ids:
            place.waypoint_ids.append(int(waypoint_id))

        previous = self.current_place_id
        self.current_place_id = place.place_id
        if link_from_current and previous and previous != place.place_id:
            self._link_adjacent(previous, place.place_id, step=step)

        self._emit('WORLD_GRAPH_PLACE', step,
                   place_id=place.place_id, created=created,
                   visit_count=place.visit_count,
                   # Position goes in the event so the merge radius is auditable
                   # from the trace: "n_places stayed at 1" has to be separable
                   # into "the agent did not move" and "merging is too greedy".
                   world_xz=[round(place.world_xz[0], 3), round(place.world_xz[1], 3)],
                   n_places=len(self.places))
        return place

    def annotate_place(self, place_id: Optional[str] = None,
                       semantic_summary: str = "",
                       todo_progress: Optional[str] = None) -> Optional[PlaceNode]:
        """Attach after-the-fact knowledge to a place the agent already stood at.

        Both fields arrive later than the place does. The room semantics come
        from the planner's own answer about the views it was just shown, which
        is one turn after ``observe_place``; the TODO id is only meaningful once
        the planner has said which item it is working on. Without this method
        ``semantic_summary`` and ``todo_progress`` stay empty, which starves the
        text side of retrieval.
        """
        place = self.places.get(place_id or self.current_place_id or '')
        if place is None:
            return None
        summary = str(semantic_summary or '').strip()
        if summary:
            place.semantic_summary = summary
        if todo_progress and todo_progress not in place.todo_progress:
            place.todo_progress.append(str(todo_progress))
        return place

    def place_text(self, place_id: str) -> str:
        """The searchable text of a place: its summary plus what was seen there.

        Lives on the graph rather than in the retriever because it walks node
        internals (``object_ids`` into the object table, and each object's
        aliases). The retriever only needs "give me words for this place".
        """
        place = self.places.get(place_id)
        if place is None:
            return ""
        parts: List[str] = []
        if place.semantic_summary:
            parts.append(place.semantic_summary)
        for obj_id in place.object_ids:
            obj = self.objects.get(obj_id)
            if obj is None:
                continue
            parts.append(obj.category)
            # The description is usually the same sentence as the category (grounding
            # writes one string), so only add it when it differs.
            if obj.description and obj.description.strip().lower() != obj.category.strip().lower():
                parts.append(obj.description)
            parts.extend(obj.aliases)
        return " ".join(p for p in parts if p)

    def distance_from(self, place_id: str, world_xz) -> Optional[float]:
        """Metres from `world_xz` to a place, or None if the place is unknown."""
        place = self.places.get(place_id)
        if place is None or world_xz is None:
            return None
        return _dist(place.world_xz, world_xz)

    def _nearest_place(self, world_xz, radius_m: float) -> Optional[PlaceNode]:
        best, best_d = None, radius_m
        for place in self.places.values():
            d = _dist(place.world_xz, world_xz)
            if d <= best_d:
                best, best_d = place, d
        return best

    def _link_adjacent(self, a: str, b: str, step: int) -> None:
        """Record that the agent walked from place `a` to place `b`."""
        pa, pb = self.places.get(a), self.places.get(b)
        if pa is None or pb is None:
            return
        if b not in pa.adjacent:
            pa.adjacent.append(b)
        if a not in pb.adjacent:
            pb.adjacent.append(a)
        self._add_edge(NAVIGABLE_TO, a, b, step=step)

    # ------------------------------------------------------------------
    # objects
    # ------------------------------------------------------------------
    def observe_object(self, category: str, step: int, description: str = "",
                       world_xz=None, map_cell=None, confidence=None,
                       bearing_deg=None, source: str = "",
                       observation_ids: Optional[List[str]] = None,
                       place_id: Optional[str] = None) -> ObjectNode:
        """Record a grounding-grounded object, merging with a prior sighting if any."""
        place_id = place_id or self.current_place_id
        existing = self._match_object(category, world_xz, place_id)
        if existing is None:
            self._object_seq += 1
            obj = ObjectNode(
                object_id=f"obj-{self._object_seq}",
                category=str(category or 'unknown'),
                description=str(description or ''),
                place_id=place_id,
                world_xz=(float(world_xz[0]), float(world_xz[1])) if world_xz is not None else None,
                map_cell=tuple(int(c) for c in map_cell) if map_cell is not None else None,
                confidence=float(confidence) if confidence is not None else None,
                bearing_deg=float(bearing_deg) if bearing_deg is not None else None,
                source=str(source or ''),
                first_seen_turn=self.turn, last_seen_turn=self.turn,
                first_seen_step=int(step), last_seen_step=int(step),
            )
            self.objects[obj.object_id] = obj
            created = True
        else:
            obj = existing
            obj.observation_count += 1
            obj.last_seen_turn = self.turn
            obj.last_seen_step = int(step)
            if description:
                obj.description = str(description)
            if world_xz is not None:
                obj.world_xz = (float(world_xz[0]), float(world_xz[1]))
            if confidence is not None:
                obj.confidence = float(confidence)
            created = False

        for obs_id in observation_ids or []:
            if obs_id not in obj.observation_ids:
                obj.observation_ids.append(obs_id)

        if not created:
            # Keep the wording that did not win. Two phrasings for one position
            # is the normal case, and which one is "the" category is arbitrary;
            # a later retrieval that matches on text needs to see both.
            cat_in = str(category or 'unknown').strip()
            if cat_in and cat_in.lower() != obj.category.strip().lower() \
                    and cat_in not in obj.aliases:
                obj.aliases.append(cat_in)

        if place_id and place_id in self.places:
            place = self.places[place_id]
            if obj.object_id not in place.object_ids:
                place.object_ids.append(obj.object_id)
            self._add_edge(CONTAINS, place_id, obj.object_id, step=step)

        self._emit('WORLD_GRAPH_OBJECT', step,
                   object_id=obj.object_id, category=obj.category,
                   created=created, place_id=place_id, n_objects=len(self.objects))
        return obj

    def _match_object(self, category: str, world_xz,
                      place_id: Optional[str]) -> Optional[ObjectNode]:
        """Find the prior sighting of this object, if there is one.

        Two rules, in order of confidence:

        1. **Same point.** Anything grounded within ``same_point_radius_m`` of an
           existing node is that node, whatever it was called. This rule exists
           because grounding writes free prose, not a label: one turn produces "the
           stainless steel stove in the kitchen" and the next "stove". Requiring
           the strings to match made ``observation_count`` permanently 1 and
           turned the object set into a log of sentences.
        2. **Same category, nearby.** A matching label within the looser
           ``object_merge_radius_m`` -- this is what absorbs back-projection
           jitter on a target the agent is walking toward.

        Unlocated groundings (no depth) can only be matched by label within the
        same place.
        """
        cat = _norm_category(category)

        if world_xz is not None:
            best, best_d = None, self.same_point_radius_m
            for obj in self.objects.values():
                if obj.world_xz is None:
                    continue
                d = _dist(obj.world_xz, world_xz)
                if d <= best_d:
                    best, best_d = obj, d
            if best is not None:
                return best

        best, best_d = None, self.object_merge_radius_m
        for obj in self.objects.values():
            if not _category_matches(obj, cat):
                continue
            if world_xz is not None and obj.world_xz is not None:
                d = _dist(obj.world_xz, world_xz)
                if d <= best_d:
                    best, best_d = obj, d
            elif world_xz is None and obj.world_xz is None and obj.place_id == place_id:
                return obj
        return best

    # ------------------------------------------------------------------
    # failures
    # ------------------------------------------------------------------
    def record_failure(self, failure_type: str, step: int, subgoal: str = "",
                       place_id: Optional[str] = None, detail: str = "",
                       object_id: Optional[str] = None,
                       covered_place_ids: Optional[List[str]] = None,
                       evidence_ids: Optional[List[str]] = None) -> Optional[FailureRecord]:
        """Attach a failed attempt and its evidence to a known place."""
        place_id = place_id or self.current_place_id
        if place_id is None or place_id not in self.places:
            # Nothing to attach to yet (failure before the first panorama).
            return None
        self._failure_seq += 1
        rec = FailureRecord(
            failure_id=f"fail-{self._failure_seq}",
            place_id=place_id,
            failure_type=str(failure_type),
            subgoal=str(subgoal or ''),
            step=int(step),
            turn=self.turn,
            detail=str(detail or ''),
            object_id=object_id,
            covered_place_ids=list(covered_place_ids or []),
            evidence_ids=list(evidence_ids or []),
        )
        self.failures[rec.failure_id] = rec
        self.places[place_id].failure_ids.append(rec.failure_id)
        self._add_edge(FAILED_FOR_SUBGOAL, place_id, rec.failure_id,
                       step=step, subgoal=rec.subgoal, failure_type=rec.failure_type)
        self._emit('WORLD_GRAPH_FAILURE', step,
                   failure_id=rec.failure_id, place_id=place_id,
                   failure_type=rec.failure_type, subgoal=rec.subgoal,
                   n_failures=len(self.failures))
        return rec

    def failures_at(self, place_id: str, subgoal: Optional[str] = None,
                    applicable_only: bool = True) -> List[FailureRecord]:
        """The failure records at a place, optionally scoped to one subgoal.

        Scoping by subgoal is deliberate: "could not reach the sofa from
        here" says nothing about looking for the TV. A place-wide count would
        make the penalty in item 3 accumulate across unrelated goals and slowly
        blacklist the whole map.
        """
        place = self.places.get(place_id)
        if place is None:
            return []
        key = (subgoal or '').strip().lower()
        out: List[FailureRecord] = []
        for fid in place.failure_ids:
            rec = self.failures.get(fid)
            if rec is None:
                continue
            if applicable_only and not rec.still_applicable:
                continue
            if key and rec.subgoal.strip().lower() != key:
                continue
            out.append(rec)
        return out

    def failure_count(self, place_id: str, subgoal: Optional[str] = None,
                      applicable_only: bool = True) -> int:
        """How many failures :meth:`failures_at` would return."""
        return len(self.failures_at(place_id, subgoal, applicable_only))

    def expire_failures(self, subgoal: Optional[str] = None,
                        place_id: Optional[str] = None) -> int:
        """Mark failures no longer applicable. Returns how many were expired.

        The counterpart to ``record_failure``: without a way to expire, an
        advisory record becomes a permanent block, which is not allowed.
        """
        n = 0
        key = (subgoal or '').strip().lower()
        for rec in self.failures.values():
            if not rec.still_applicable:
                continue
            if place_id and rec.place_id != place_id:
                continue
            if key and rec.subgoal.strip().lower() != key:
                continue
            rec.still_applicable = False
            n += 1
        return n

    # ------------------------------------------------------------------
    # edges / summaries
    # ------------------------------------------------------------------
    def _add_edge(self, edge_type: str, src: str, dst: str, **fields: Any) -> None:
        # NAVIGABLE_TO is symmetric -- `_link_adjacent` already writes both
        # `adjacent` lists -- so it is keyed on the unordered pair. Keying it on
        # the ordered one would log a second edge the first time the agent walks
        # back down a corridor it already knows, and inflate every downstream
        # count. CONTAINS and OBSERVED_FROM are directional and keep their order.
        if edge_type == NAVIGABLE_TO:
            key = (edge_type,) + tuple(sorted((src, dst)))
        else:
            key = (edge_type, src, dst)
        if edge_type in (NAVIGABLE_TO, CONTAINS, OBSERVED_FROM) and key in self._edge_index:
            return
        self._edge_index.add(key)
        edge = {'type': edge_type, 'src': src, 'dst': dst}
        edge.update({k: v for k, v in fields.items() if v is not None})
        self.edges.append(edge)

    def neighbours(self, place_id: str) -> List[str]:
        place = self.places.get(place_id)
        return list(place.adjacent) if place else []

    def edge_counts(self) -> Dict[str, int]:
        counts = {t: 0 for t in EDGE_TYPES}
        for e in self.edges:
            counts[e['type']] = counts.get(e['type'], 0) + 1
        return counts

    def failure_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rec in self.failures.values():
            counts[rec.failure_type] = counts.get(rec.failure_type, 0) + 1
        return counts

    def stats(self) -> Dict[str, Any]:
        """Per-episode aggregate for the trace and for report_rounds."""
        visits = [p.visit_count for p in self.places.values()] or [0]
        return {
            'schema': SCHEMA_WORLD_GRAPH,
            'n_places': len(self.places),
            'n_objects': len(self.objects),
            'n_failures': len(self.failures),
            'n_edges': len(self.edges),
            'edges_by_type': self.edge_counts(),
            'failures_by_type': self.failure_type_counts(),
            'max_visit_count': max(visits),
            'revisited_places': sum(1 for v in visits if v > 1),
            'current_place_id': self.current_place_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema': SCHEMA_WORLD_GRAPH,
            'places': [p.summary() for p in self.places.values()],
            'objects': [o.summary() for o in self.objects.values()],
            'failures': [f.summary() for f in self.failures.values()],
            'edges': list(self.edges),
        }

    def _emit(self, event_type: str, step: int, **fields: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event_type, int(step), **fields)
        except Exception:
            pass


def _dist(a, b) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _norm_category(category: Any) -> str:
    return str(category or 'unknown').strip().lower()


def _category_matches(obj: ObjectNode, cat: str) -> bool:
    """Whether `cat` names `obj`, allowing for the aliases it has been called."""
    if _norm_category(obj.category) == cat:
        return True
    return any(_norm_category(a) == cat for a in obj.aliases)
