"""Live tests for the bi-temporal graph layer.

Requires a FalkorDB reachable at localhost:6390. Tests are skipped if
the database is not available so the suite stays green in CI without
infrastructure.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rke.config import DEFAULTS, Config
from rke.graph_store import GraphStore
from rke.graph_temporal import TemporalGraphStore, TemporalRelation

UTC = timezone.utc


def _uid(name: str) -> str:
    """Namespaced identifier so each test owns its own nodes."""
    return f"test_graph_temporal::{name}"


@pytest.fixture
def graph():
    cfg = Config(raw={
        **DEFAULTS,
        "falkordb": {
            "host": "localhost",
            "port": 6390,
            "graph": "rke_temporal_test",
            "password": None,
        },
    })
    gs = GraphStore(cfg)
    if not gs.ping():
        pytest.skip("FalkorDB not reachable on localhost:6390")
    gs.query("MATCH (n) DETACH DELETE n")
    return gs


@pytest.fixture
def tstore(graph):
    return TemporalGraphStore(graph)


def test_naive_datetime_rejected(tstore):
    rel = TemporalRelation(
        src_label="Person",
        src_name=_uid("naive_src"),
        rel_type="USES",
        dst_label="Tool",
        dst_name=_uid("naive_dst"),
        valid_from=datetime(2024, 1, 1),  # naive
    )
    with pytest.raises(ValueError):
        tstore.add_relation(rel)


def test_add_relation_persists_temporal_fields(tstore, graph):
    vf = datetime(2024, 1, 1, tzinfo=UTC)
    vt = datetime(2024, 6, 1, tzinfo=UTC)
    rec = datetime(2024, 1, 2, tzinfo=UTC)
    src = _uid("persist_src")
    dst = _uid("persist_dst")
    tstore.add_relation(TemporalRelation(
        src_label="Person",
        src_name=src,
        rel_type="USES",
        dst_label="Tool",
        dst_name=dst,
        valid_from=vf,
        valid_to=vt,
        recorded_at=rec,
        properties={"confidence": 0.9},
    ))
    rows = graph.query(
        "MATCH (a:Person {name: $s})-[r:USES]->(b:Tool {name: $d}) "
        "RETURN r.valid_from, r.valid_to, r.recorded_at, r.confidence",
        {"s": src, "d": dst},
    )
    assert len(rows) == 1
    assert rows[0][0] == vf.isoformat()
    assert rows[0][1] == vt.isoformat()
    assert rows[0][2] == rec.isoformat()
    assert rows[0][3] == 0.9


def test_query_at_between_two_relations(tstore):
    src = _uid("between_src")
    dst_a = _uid("between_dst_a")
    dst_b = _uid("between_dst_b")
    t1 = datetime(2024, 1, 1, tzinfo=UTC)
    t2 = datetime(2024, 3, 1, tzinfo=UTC)
    # Phase 1: src USES dst_a
    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", dst_a, valid_from=t1, valid_to=t2,
    ))
    # Phase 2: src USES dst_b (open)
    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", dst_b, valid_from=t2,
    ))

    # At t = mid-phase-1, only dst_a is valid
    mid1 = datetime(2024, 2, 1, tzinfo=UTC)
    rows = tstore.query_at(
        "MATCH (a:Person {name: $s})-[r:USES]->(b:Tool) RETURN b.name",
        mid1,
        {"s": src},
    )
    names = {r[0] for r in rows}
    assert names == {dst_a}

    # At t = mid-phase-2, only dst_b is valid
    mid2 = datetime(2024, 4, 1, tzinfo=UTC)
    rows = tstore.query_at(
        "MATCH (a:Person {name: $s})-[r:USES]->(b:Tool) RETURN b.name",
        mid2,
        {"s": src},
    )
    names = {r[0] for r in rows}
    assert names == {dst_b}


def test_query_at_after_closed_relation_excluded(tstore):
    src = _uid("closed_src")
    dst = _uid("closed_dst")
    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", dst,
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        valid_to=datetime(2024, 2, 1, tzinfo=UTC),
    ))
    rows = tstore.query_at(
        "MATCH (a:Person {name: $s})-[r:USES]->(b:Tool) RETURN b.name",
        datetime(2024, 3, 1, tzinfo=UTC),
        {"s": src},
    )
    assert rows == []


def test_invalidate_sets_valid_to_and_is_idempotent(tstore, graph):
    src = _uid("inval_src")
    dst = _uid("inval_dst")
    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", dst,
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
    ))
    close_at = datetime(2024, 5, 1, tzinfo=UTC)
    updated = tstore.invalidate(
        "Person", src, "USES", "Tool", dst, at=close_at,
    )
    assert updated == 1

    rows = graph.query(
        "MATCH (a:Person {name: $s})-[r:USES]->(b:Tool {name: $d}) "
        "RETURN r.valid_to",
        {"s": src, "d": dst},
    )
    assert rows[0][0] == close_at.isoformat()

    # Second invalidate must not find an open edge and must not
    # overwrite the stored timestamp with a later one.
    later = datetime(2024, 9, 1, tzinfo=UTC)
    updated_again = tstore.invalidate(
        "Person", src, "USES", "Tool", dst, at=later,
    )
    assert updated_again == 0
    rows = graph.query(
        "MATCH (a:Person {name: $s})-[r:USES]->(b:Tool {name: $d}) "
        "RETURN r.valid_to",
        {"s": src, "d": dst},
    )
    assert rows[0][0] == close_at.isoformat()


def test_history_returns_all_versions_sorted(tstore):
    src = _uid("hist_src")
    dst = _uid("hist_dst")
    t1 = datetime(2023, 1, 1, tzinfo=UTC)
    t2 = datetime(2023, 6, 1, tzinfo=UTC)
    t3 = datetime(2024, 1, 1, tzinfo=UTC)
    t4 = datetime(2024, 6, 1, tzinfo=UTC)
    # Insert in non-chronological order to exercise the ORDER BY
    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", dst, valid_from=t3, valid_to=t4,
    ))
    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", dst, valid_from=t1, valid_to=t2,
    ))
    hist = tstore.history("Person", src, "USES", "Tool", dst)
    assert len(hist) == 2
    assert hist[0]["valid_from"] == t1.isoformat()
    assert hist[0]["valid_to"] == t2.isoformat()
    assert hist[1]["valid_from"] == t3.isoformat()
    assert hist[1]["valid_to"] == t4.isoformat()


def test_fact_changes_between_filters_by_valid_from(tstore):
    src = _uid("changes_src")
    dst1 = _uid("changes_dst1")
    dst2 = _uid("changes_dst2")
    dst3 = _uid("changes_dst3")
    before = datetime(2024, 1, 1, tzinfo=UTC)
    inside_a = datetime(2024, 3, 1, tzinfo=UTC)
    inside_b = datetime(2024, 4, 1, tzinfo=UTC)
    after = datetime(2024, 7, 1, tzinfo=UTC)

    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", dst1, valid_from=before,
    ))
    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", dst2, valid_from=inside_a,
    ))
    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", dst3, valid_from=inside_b,
    ))
    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", _uid("changes_dst4"), valid_from=after,
    ))

    window_start = datetime(2024, 2, 1, tzinfo=UTC)
    window_end = datetime(2024, 6, 1, tzinfo=UTC)
    changes = tstore.fact_changes_between(window_start, window_end)
    dst_names = {c["dst_name"] for c in changes if c["src_name"] == src}
    assert dst_names == {dst2, dst3}


def test_query_at_handles_open_ended(tstore):
    src = _uid("open_src")
    dst = _uid("open_dst")
    tstore.add_relation(TemporalRelation(
        "Person", src, "USES", "Tool", dst,
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        # valid_to left None → open-ended
    ))
    # A timestamp a few years out should still see the fact as valid.
    far_future = datetime(2099, 1, 1, tzinfo=UTC)
    rows = tstore.query_at(
        "MATCH (a:Person {name: $s})-[r:USES]->(b:Tool) RETURN b.name",
        far_future,
        {"s": src},
    )
    assert {r[0] for r in rows} == {dst}

    # Also correct right at valid_from (inclusive).
    at_start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = tstore.query_at(
        "MATCH (a:Person {name: $s})-[r:USES]->(b:Tool) RETURN b.name",
        at_start,
        {"s": src},
    )
    assert {r[0] for r in rows} == {dst}

    # Before valid_from should exclude it.
    before = datetime(2023, 12, 31, tzinfo=UTC)
    rows = tstore.query_at(
        "MATCH (a:Person {name: $s})-[r:USES]->(b:Tool) RETURN b.name",
        before,
        {"s": src},
    )
    assert rows == []


def test_fact_changes_between_rejects_naive(tstore):
    with pytest.raises(ValueError):
        tstore.fact_changes_between(
            datetime(2024, 1, 1),  # naive
            datetime(2024, 2, 1, tzinfo=UTC),
        )
