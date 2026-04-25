"""Bi-temporal layer over :class:`rke.graph_store.GraphStore`.

Models Zep-style bi-temporal facts: every relation carries

- ``valid_from`` — when the fact became true in the world (inclusive, UTC)
- ``valid_to``   — when the fact stopped being true (exclusive, UTC, or open)
- ``recorded_at`` — when we learned the fact (defaults to insert-time UTC)

Timestamps are persisted as ISO-8601 strings in the relation properties,
which FalkorDB stores natively. Lexicographic comparison on tz-aware
ISO strings with a common offset (UTC) is order-preserving, so range
queries can be expressed directly in Cypher.

The store is *composable*: it wraps an existing ``GraphStore`` rather than
subclassing it, so callers can mix the plain and temporal APIs freely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .graph_store import GraphStore, _safe_label


def _require_utc(name: str, dt: datetime | None) -> datetime | None:
    """Reject naive datetimes; normalise aware datetimes to UTC."""
    if dt is None:
        return None
    if not isinstance(dt, datetime):
        raise ValueError(f"{name} must be a datetime, got {type(dt).__name__}")
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"{name} must be timezone-aware (UTC required)")
    return dt.astimezone(timezone.utc)


def _to_iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TemporalRelation:
    src_label: str
    src_name: str
    rel_type: str
    dst_label: str
    dst_name: str
    valid_from: datetime
    valid_to: datetime | None = None
    recorded_at: datetime | None = None
    properties: dict[str, Any] = field(default_factory=dict)


class TemporalGraphStore:
    """Bi-temporal facade over a :class:`GraphStore`."""

    # Match the final RETURN keyword (case-insensitive, word-bounded).
    _RETURN_RE = re.compile(r"\bRETURN\b", re.IGNORECASE)

    def __init__(self, base: GraphStore) -> None:
        self.base = base

    # ── Mutations ────────────────────────────────────────────────
    def add_relation(self, rel: TemporalRelation) -> None:
        """Persist a bi-temporal relation.

        Uses ``CREATE`` (not ``MERGE``) so multiple historical versions of
        the same (src, rel_type, dst) can coexist as distinct edges.
        """
        vf = _require_utc("valid_from", rel.valid_from)
        vt = _require_utc("valid_to", rel.valid_to)
        rec = _require_utc("recorded_at", rel.recorded_at) or _now_utc()
        if vf is None:
            raise ValueError("valid_from is required")

        props: dict[str, Any] = dict(rel.properties or {})
        props["valid_from"] = _to_iso(vf)
        if vt is not None:
            props["valid_to"] = _to_iso(vt)
        props["recorded_at"] = _to_iso(rec)

        cypher = (
            f"MERGE (a:{_safe_label(rel.src_label)} {{name: $sname}}) "
            f"MERGE (b:{_safe_label(rel.dst_label)} {{name: $dname}}) "
            f"CREATE (a)-[r:{_safe_label(rel.rel_type)}]->(b) "
            f"SET r += $props"
        )
        self.base.graph.query(cypher, {
            "sname": rel.src_name,
            "dname": rel.dst_name,
            "props": props,
        })

    def invalidate(
        self,
        src_label: str,
        src_name: str,
        rel_type: str,
        dst_label: str,
        dst_name: str,
        *,
        at: datetime | None = None,
    ) -> int:
        """Close the open version of a relation by setting ``valid_to``.

        Idempotent: only targets edges where ``valid_to`` is null/missing,
        so a second call is a no-op.
        Returns the number of rows updated.
        """
        at_utc = _require_utc("at", at) or _now_utc()
        cypher = (
            f"MATCH (a:{_safe_label(src_label)} {{name: $sname}})"
            f"-[r:{_safe_label(rel_type)}]->"
            f"(b:{_safe_label(dst_label)} {{name: $dname}}) "
            "WHERE r.valid_to IS NULL "
            "SET r.valid_to = $at "
            "RETURN count(r)"
        )
        rows = self.base.query(cypher, {
            "sname": src_name,
            "dname": dst_name,
            "at": _to_iso(at_utc),
        })
        return int(rows[0][0]) if rows else 0

    # ── Queries ──────────────────────────────────────────────────
    def query_at(
        self,
        cypher_match: str,
        t: datetime,
        params: dict | None = None,
    ) -> list[list[Any]]:
        """Run a caller-supplied query constrained to facts valid at ``t``.

        The query must alias the relation as ``r`` and contain a single
        final ``RETURN`` clause. The temporal predicate is injected as a
        ``WITH * WHERE ...`` step immediately before ``RETURN``.
        """
        t_utc = _require_utc("t", t)
        match = self._RETURN_RE.search(cypher_match)
        if match is None:
            raise ValueError("cypher_match must contain a RETURN clause")

        head = cypher_match[: match.start()].rstrip()
        tail = cypher_match[match.start():]
        temporal = (
            "WITH * WHERE r.valid_from <= $_t_at "
            "AND (r.valid_to IS NULL OR r.valid_to > $_t_at) "
        )
        merged = f"{head} {temporal}{tail}"

        merged_params: dict[str, Any] = dict(params or {})
        merged_params["_t_at"] = _to_iso(t_utc)
        return self.base.query(merged, merged_params)

    def history(
        self,
        src_label: str,
        src_name: str,
        rel_type: str,
        dst_label: str,
        dst_name: str,
    ) -> list[dict[str, Any]]:
        """All historical versions of a relation, oldest first."""
        cypher = (
            f"MATCH (a:{_safe_label(src_label)} {{name: $sname}})"
            f"-[r:{_safe_label(rel_type)}]->"
            f"(b:{_safe_label(dst_label)} {{name: $dname}}) "
            "RETURN r.valid_from, r.valid_to, r.recorded_at, properties(r) "
            "ORDER BY r.valid_from ASC"
        )
        rows = self.base.query(cypher, {"sname": src_name, "dname": dst_name})
        return [
            {
                "valid_from": r[0],
                "valid_to": r[1],
                "recorded_at": r[2],
                "properties": r[3],
            }
            for r in rows
        ]

    def fact_changes_between(
        self,
        t0: datetime,
        t1: datetime,
    ) -> list[dict[str, Any]]:
        """Relations whose ``valid_from`` falls in ``[t0, t1)``."""
        t0_utc = _require_utc("t0", t0)
        t1_utc = _require_utc("t1", t1)
        cypher = (
            "MATCH (a)-[r]->(b) "
            "WHERE r.valid_from >= $t0 AND r.valid_from < $t1 "
            "RETURN a.name, type(r), b.name, "
            "       r.valid_from, r.valid_to, r.recorded_at "
            "ORDER BY r.valid_from ASC"
        )
        rows = self.base.query(cypher, {
            "t0": _to_iso(t0_utc),
            "t1": _to_iso(t1_utc),
        })
        return [
            {
                "src_name": r[0],
                "rel_type": r[1],
                "dst_name": r[2],
                "valid_from": r[3],
                "valid_to": r[4],
                "recorded_at": r[5],
            }
            for r in rows
        ]
