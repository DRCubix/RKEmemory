"""End-to-end smoke test against live Qdrant + FalkorDB.

Run with the test instances:
    RKE_QDRANT_PORT=6433 RKE_QDRANT_GRPC_PORT=6434 \
    RKE_FALKORDB_PORT=6390 \
    RKE_QDRANT_COLLECTION=rke_smoke \
    RKE_FALKORDB_GRAPH=rke_smoke \
    RKE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    RKE_EMBEDDING_DIMENSIONS=384 \
    RKE_WIKI_PATH=/tmp/rke_smoke_wiki \
    python scripts/e2e_smoke.py
"""

from __future__ import annotations

import shutil
import sys

from rke.agent_integration import format_context_for_agent, gather_context
from rke.config import load_config
from rke.graph_store import Entity, GraphStore, Relation
from rke.rlm.router import RLMRouter
from rke.vector_store import VectorStore
from rke.wiki.knowledge_base import KnowledgeBase
from rke.wiki.manager import WikiManager


def banner(label: str) -> None:
    print(f"\n=== {label} ===")


def main() -> int:
    cfg = load_config()
    print(f"wiki path : {cfg.wiki_path}")
    print(f"qdrant    : {cfg.qdrant['host']}:{cfg.qdrant['port']} (collection={cfg.qdrant['collection']})")
    print(f"falkordb  : {cfg.falkordb['host']}:{cfg.falkordb['port']} (graph={cfg.falkordb['graph']})")
    print(f"embedding : {cfg.embedding['model']} ({cfg.embedding['dimensions']}d) on {cfg.embedding['device']}")

    # Clean any leftover wiki dir from a prior run.
    if cfg.wiki_path.exists():
        shutil.rmtree(cfg.wiki_path)

    failures: list[str] = []

    # ── 1. Wiki + KnowledgeBase ──────────────────────────────────
    banner("1. Wiki page create + index")
    kb = KnowledgeBase(cfg)
    page1 = kb.add_page(
        title="OAuth Token Switching",
        body="When users switch between accounts, the OAuth refresh token must be rotated and stored per-user.",
        category="entities",
        tags=["auth", "oauth"],
    )
    kb.add_page(
        title="Postgres Tuning",
        body="Increase work_mem for analytical queries; tune shared_buffers based on RAM.",
        category="entities",
        tags=["db", "postgres"],
    )
    pages = WikiManager(cfg).list_pages()
    if len(pages) != 2:
        failures.append(f"expected 2 pages, got {len(pages)}")
    print(f"  pages on disk: {[p.slug for p in pages]}")
    print(f"  page1.path exists: {page1.path.exists()}")

    # ── 2. Vector search ─────────────────────────────────────────
    banner("2. Vector search via Qdrant")
    vs = VectorStore(cfg)
    info = vs.collection_info()
    print(f"  collection info: {info}")
    if "error" in info:
        failures.append(f"qdrant collection error: {info['error']}")

    hits = vs.search("oauth refresh tokens for accounts", limit=3)
    print(f"  hits for 'oauth refresh tokens': {len(hits)}")
    if not hits:
        failures.append("vector search returned no hits")
    else:
        top = hits[0]
        print(f"  top hit: score={top.score:.3f} title={top.metadata.get('title')!r}")
        print(f"           text={top.text[:80]!r}")
        if top.metadata.get("title") != "OAuth Token Switching":
            failures.append(f"expected OAuth page top hit, got {top.metadata.get('title')!r}")

    # Hybrid search should still work and rank the right doc.
    hhits = vs.hybrid_search("postgres work_mem tuning", limit=2)
    print(f"  hybrid hits for 'postgres work_mem': {len(hhits)}")
    if hhits and hhits[0].metadata.get("title") != "Postgres Tuning":
        failures.append(f"hybrid: expected Postgres page top, got {hhits[0].metadata.get('title')!r}")

    # ── 3. FalkorDB graph ────────────────────────────────────────
    banner("3. Knowledge graph (FalkorDB)")
    gs = GraphStore(cfg)
    if not gs.ping():
        failures.append("falkordb ping failed")
    else:
        gs.add_entity(Entity("Project", "RKE", {"description": "Recursive Knowledge Engine"}))
        gs.add_entity(Entity("API", "Qdrant", {"kind": "vector_db"}))
        gs.add_entity(Entity("API", "FalkorDB", {"kind": "graph_db"}))
        gs.add_relation(Relation("Project", "RKE", "USES", "API", "Qdrant"))
        gs.add_relation(Relation("Project", "RKE", "USES", "API", "FalkorDB"))
        stats = gs.stats()
        print(f"  graph stats: {stats}")
        if stats.get("nodes", 0) < 3:
            failures.append(f"expected >=3 nodes, got {stats.get('nodes')}")
        rows = gs.query("MATCH (p:Project)-[:USES]->(a:API) RETURN p.name, a.name")
        print(f"  cypher rows: {rows}")
        if not rows:
            failures.append("cypher MATCH returned no rows")

    # ── 4. Combined query (KB) ───────────────────────────────────
    banner("4. KnowledgeBase combined query")
    combined = kb.query("oauth refresh tokens", wiki_limit=2, vector_limit=3)
    print(f"  combined hits: {len(combined)}")
    for h in combined[:3]:
        print(f"    [{h.source}] score={h.score:.2f} {h.title!r}")
    if not combined:
        failures.append("combined query returned nothing")

    # ── 5. Agent context block ───────────────────────────────────
    banner("5. gather_context + format_context_for_agent")
    ctx = gather_context("How does oauth token switching work?", config=cfg)
    out = format_context_for_agent(ctx, agent="claude")
    print(out)
    if "<context>" not in out or "OAuth Token Switching" not in out:
        failures.append("agent context block missing expected content")

    # ── 6. RLM router (deterministic mode) ───────────────────────
    banner("6. RLM router (deterministic fallback)")
    router = RLMRouter(config=cfg)
    res = router.complete("oauth refresh tokens")
    print(f"  used_llm={res.used_llm} iterations={res.iterations}")
    print(f"  answer (first 300):\n{res.answer[:300]}")
    if res.used_llm:
        failures.append("expected deterministic mode (no LLM provider)")
    if "OAuth Token Switching" not in res.answer:
        failures.append("RLM answer missing expected wiki hit")

    # ── 7. Lint ──────────────────────────────────────────────────
    banner("7. Wiki lint")
    findings = WikiManager(cfg).lint()
    print(f"  findings: {len(findings)}")
    for f in findings[:5]:
        print(f"    {f.kind} {f.page}: {f.detail}")

    # ── Summary ──────────────────────────────────────────────────
    banner("RESULT")
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK — all 7 stages passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
