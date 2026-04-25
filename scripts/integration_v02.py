"""End-to-end integration test for v0.2 features.

Wires all 5 features together and proves they cooperate:
  1. Inverted index (search_index)
  2. LLM extraction (extractor, regex backend)
  3. Lifecycle / TTL
  4. Bi-temporal graph
  5. Chat memory adapter

Requires live Qdrant + FalkorDB at the configured ports.
"""

from __future__ import annotations

import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from rke.config import load_config
from rke.graph_store import GraphStore
from rke.vector_store import VectorStore
from rke.wiki.manager import WikiManager, clear_hooks

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, ok: bool, det: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append((name, det))
        print(f"  [FAIL] {name}: {det}")


def main() -> int:
    cfg = load_config()
    print(f"=== v0.2 INTEGRATION ===  qdrant={cfg.qdrant['port']}  falkordb={cfg.falkordb['port']}  wiki={cfg.wiki_path}")
    if cfg.wiki_path.exists():
        shutil.rmtree(cfg.wiki_path)
    clear_hooks()

    # ── Feature 1: Inverted index attached ─────────────────────
    print("\n--- 1. Inverted index ---")
    from rke.wiki.search_index import WhooshIndex  # noqa: E402
    wm = WikiManager(cfg)
    idx_dir = Path("/tmp/rke_int_idx")
    if idx_dir.exists():
        shutil.rmtree(idx_dir)
    WhooshIndex.attach(wm, idx_dir)

    # ── Feature 2: LLM extractor (regex backend) attached ──────
    print("--- 2. Extractor (regex backend) attached ---")
    from rke.knowledge.extractor import EntityExtractor, attach_to_wiki  # noqa: E402
    fake_graph = MagicMock(spec=GraphStore)
    fake_graph.add_entity = MagicMock()
    fake_graph.add_relation = MagicMock()
    attach_to_wiki(fake_graph, extractor=EntityExtractor(provider="regex"))

    # Create some pages
    wm.create_page("OAuth Patterns", "Project Alpha uses Qdrant. Qdrant depends on Rocksdb.", "entities", ["auth"])
    wm.create_page("Postgres Tuning", "tune work_mem and shared_buffers. Use Postgres carefully.", "entities", ["db"])
    wm.create_page("Vector Search", "Cosine similarity over normalized embeddings is fast.", "notes", ["vector"])
    time.sleep(0.2)

    hits = wm.query_wiki("oauth")
    check("Inverted index returns OAuth page", bool(hits) and hits[0].slug == "oauth-patterns")

    hits2 = wm.query_wiki("postgres tuning")
    check("Inverted index ranks Postgres page", bool(hits2) and hits2[0].slug == "postgres-tuning")

    # Extractor should have fired on the 3 creates → at least some entities
    check("Extractor invoked add_entity", fake_graph.add_entity.call_count >= 1,
          f"add_entity called {fake_graph.add_entity.call_count} times")

    # ── Feature 3: Lifecycle ───────────────────────────────────
    print("\n--- 3. Lifecycle / TTL ---")
    from rke.wiki.lifecycle import (  # noqa: E402
        evict_expired,
        is_expired,
        set_expiry,
        touch,
    )
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    p = set_expiry(wm, "vector-search", at=past)
    check("set_expiry returns updated page", p is not None and p.expires_at)
    p2 = wm.get_page("vector-search")
    check("expires_at persisted to disk", p2 is not None and p2.expires_at != "")
    check("is_expired detects past timestamp", p2 is not None and is_expired(p2))
    evicted = evict_expired(wm)
    check("evict_expired removes page", "vector-search" in evicted)
    check("evicted page is gone from index", wm.get_page("vector-search") is None)

    # touch updates last_accessed_at
    t = touch(wm, "oauth-patterns")
    check("touch sets last_accessed_at", t is not None and t.last_accessed_at != "")

    # ── Feature 4: Bi-temporal graph ───────────────────────────
    print("\n--- 4. Bi-temporal graph ---")
    from rke.graph_store import Entity  # noqa: E402
    from rke.graph_temporal import TemporalGraphStore, TemporalRelation  # noqa: E402
    gs = GraphStore(cfg)
    gs.connect()
    gs.query("MATCH (n) WHERE n.name STARTS WITH 'IntTest_' DETACH DELETE n")
    tgs = TemporalGraphStore(gs)
    gs.add_entity(Entity("Project", "IntTest_App"))
    gs.add_entity(Entity("API", "IntTest_DB_v1"))
    gs.add_entity(Entity("API", "IntTest_DB_v2"))

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2025, 6, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tgs.add_relation(TemporalRelation(
        "Project", "IntTest_App", "USES", "API", "IntTest_DB_v1",
        valid_from=t0, valid_to=t1,
    ))
    tgs.add_relation(TemporalRelation(
        "Project", "IntTest_App", "USES", "API", "IntTest_DB_v2",
        valid_from=t1,
    ))
    rows_at_q1 = tgs.query_at(
        "MATCH (p:Project {name: $sn})-[r:USES]->(a:API) RETURN a.name",
        datetime(2025, 3, 1, tzinfo=timezone.utc),
        params={"sn": "IntTest_App"},
    )
    check("query_at(early) returns DB_v1", any("IntTest_DB_v1" in str(row) for row in rows_at_q1))
    rows_at_q2 = tgs.query_at(
        "MATCH (p:Project {name: $sn})-[r:USES]->(a:API) RETURN a.name",
        t2,
        params={"sn": "IntTest_App"},
    )
    check("query_at(late) returns DB_v2", any("IntTest_DB_v2" in str(row) for row in rows_at_q2))

    # ── Feature 5: Chat memory ─────────────────────────────────
    print("\n--- 5. Chat memory adapter ---")
    from rke.adapters.chat_memory import ChatMemory  # noqa: E402
    cm = ChatMemory(thread_id="int-test-thread", config=cfg, buffer_size=3, summarize_threshold=5)
    cm.add_user_message("Hello")
    cm.add_assistant_message("Hi there")
    cm.add_user_message("What's our auth setup?")
    msgs = cm.history()
    check("Chat memory stores 3 messages", len(msgs) == 3)
    check("Chat memory chronological order", msgs[0].role == "user" and msgs[2].content.startswith("What"))
    prompt = cm.to_prompt_string()
    check("to_prompt_string contains both roles", "user:" in prompt.lower() and "assistant:" in prompt.lower())

    # Persistence round-trip
    cm2 = ChatMemory(thread_id="int-test-thread", config=cfg, buffer_size=3, summarize_threshold=5)
    msgs2 = cm2.history()
    check("Chat memory persists to disk", len(msgs2) == 3 and msgs2[0].content == "Hello")

    # KB-backed mode: writes route through KnowledgeBase → vector index
    # stays in sync. Use a UUID sentinel so we can't accidentally satisfy
    # the assertion via stale state, and clear the relevant Qdrant points
    # first. The assertion specifically requires a VECTOR hit (not just
    # a wiki keyword hit) to prove the routing.
    import uuid as _uuid

    from rke.wiki.knowledge_base import KnowledgeBase  # noqa: E402
    sentinel = f"sentinel{_uuid.uuid4().hex[:12]}"
    kb_for_chat = KnowledgeBase(cfg)
    # Drop any prior chunks for this thread slug so we know any vector hit
    # came from THIS run's writes.
    try:
        kb_for_chat.vectors.delete_by_filter({"slug": "kb-thread-int"})
    except Exception:
        pass
    cm_kb = ChatMemory(thread_id="kb-thread-int", kb=kb_for_chat,
                       buffer_size=3, summarize_threshold=5)
    cm_kb.add_user_message(f"Discuss {sentinel} in this conversation")
    cm_kb.add_assistant_message(f"Yes, {sentinel} is configured")
    time.sleep(0.5)
    long_term = cm_kb.search_long_term(sentinel, limit=5)
    vector_hits = [h for h in long_term if h.source == "vector"]
    check(
        "KB-backed ChatMemory routes to vector index (vector hit present)",
        bool(vector_hits),
        f"got {len(long_term)} total hits, {len(vector_hits)} vector hits",
    )

    # Vector search still works (not regressed)
    print("\n--- Regression check: vector search ---")
    vs = VectorStore(cfg)
    vs.ensure_collection()
    ids = vs.upsert(["regression sample about machine learning"], metadatas=[{"src": "regression"}])
    time.sleep(0.2)
    hits = vs.search("machine learning", limit=1)
    info = vs.collection_info()
    check("Qdrant collection healthy after v0.2 features", "error" not in info)
    check("Vector search still returns hits", bool(hits))
    vs.delete(ids)

    # ── Summary ──
    print(f"\n=== INTEGRATION RESULT: {len(PASSED)} pass, {len(FAILED)} fail ===")
    if FAILED:
        for n, d in FAILED:
            print(f"  [FAIL] {n}: {d}")
        return 1
    print("ALL v0.2 FEATURES INTEGRATED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
