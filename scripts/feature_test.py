"""Comprehensive feature test — exercises every public API.

Pass/fail per stage; exits non-zero on any failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rke import __version__
from rke.agent_integration import format_context_for_agent, gather_context
from rke.config import DEFAULTS, _coerce, _deep_merge, load_config
from rke.graph_store import Entity, GraphStore, Relation
from rke.ingestion.git_repos import list_repos
from rke.ingestion.knowledge import iter_files
from rke.rlm.environment import Environment, make_environment
from rke.rlm.router import RLMRouter
from rke.vector_store import VectorStore
from rke.wiki.knowledge_base import KnowledgeBase, chunk_text
from rke.wiki.manager import WikiManager, slugify

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append((name, detail))
        print(f"  [FAIL] {name}: {detail}")


def section(label: str) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")


def main() -> int:
    cfg = load_config()
    print(f"rke {__version__}")
    print(f"qdrant   = {cfg.qdrant['host']}:{cfg.qdrant['port']}")
    print(f"falkordb = {cfg.falkordb['host']}:{cfg.falkordb['port']}")
    print(f"wiki     = {cfg.wiki_path}")
    print(f"embed    = {cfg.embedding['model']} ({cfg.embedding['dimensions']}d)")

    if cfg.wiki_path.exists():
        shutil.rmtree(cfg.wiki_path)

    # ── 1. Config ────────────────────────────────────────────────
    section("1. CONFIG (layered loader, env overrides, coercion)")
    check("DEFAULTS has all sections", all(k in DEFAULTS for k in ["wiki", "qdrant", "falkordb", "embedding", "rlm", "ingestion", "logging"]))
    merged = _deep_merge({"a": {"x": 1}}, {"a": {"y": 2}})
    check("deep_merge recursive", merged == {"a": {"x": 1, "y": 2}})
    check("_coerce booleans", _coerce("true") is True and _coerce("false") is False)
    check("_coerce numerics", _coerce("42") == 42 and _coerce("3.14") == 3.14)
    # Inject a unique sentinel env var, reload, and verify env > yaml > defaults
    # precedence — robust regardless of any user-local config/rke.yaml.
    sentinel = "feature-test-sentinel-xyz"
    os.environ["RKE_FALKORDB__GRAPH"] = sentinel
    try:
        cfg2 = load_config()
        check(
            "env override applied (sentinel through layered loader)",
            cfg2.falkordb["graph"] == sentinel,
            f"got falkordb.graph={cfg2.falkordb['graph']!r}",
        )
    finally:
        del os.environ["RKE_FALKORDB__GRAPH"]
    check("config.wiki_path is Path", isinstance(cfg.wiki_path, Path))

    # ── 2. Wiki ──────────────────────────────────────────────────
    section("2. WIKI MANAGER (slugify, CRUD, query, lint)")
    check("slugify normalizes", slugify("My Topic!") == "my-topic")
    check("slugify empty", slugify("") == "untitled")
    wm = WikiManager(cfg)
    p = wm.create_page("OAuth Patterns", "# OAuth\n\nrefresh tokens for accounts", "entities", ["auth"])
    check("create_page wrote file", p.path is not None and p.path.exists())
    fetched = wm.get_page("OAuth Patterns")
    check("get_page round-trips", fetched is not None and fetched.title == "OAuth Patterns")
    wm.create_page("OAuth Patterns", "# v2\n\nupdated", "entities")
    refetched = wm.get_page("OAuth Patterns")
    check("update preserves tags", refetched is not None and refetched.tags == ["auth"])
    check("update overwrites body", refetched is not None and "updated" in refetched.body)
    wm.create_page("Postgres", "tune work_mem", "entities", ["db"])
    wm.create_page("Empty", "", "general", [])
    hits = wm.query_wiki("postgres")
    check("query_wiki keyword", bool(hits) and hits[0].title == "Postgres")
    findings = wm.lint()
    check("lint finds empty page", any(f.kind == "empty" for f in findings))
    pages_n = len(wm.list_pages())
    check("list_pages count", pages_n == 3, f"got {pages_n}")
    check("delete_page removes file", wm.delete_page("Empty"))

    # ── 3. Vector store ──────────────────────────────────────────
    section("3. VECTOR STORE (Qdrant + embed + search + hybrid)")
    vs = VectorStore(cfg)
    vs.ensure_collection()
    info = vs.collection_info()
    check("collection ensured", "error" not in info)
    embeddings = vs.embed(["hello world", "second doc"])
    check("embed returns vectors", len(embeddings) == 2 and len(embeddings[0]) == cfg.embedding["dimensions"])
    ids = vs.upsert(["alpha document about birds", "beta document about cars"], metadatas=[{"k": "a"}, {"k": "b"}])
    check("upsert returns ids", len(ids) == 2)
    time.sleep(0.3)
    res = vs.search("flying birds", limit=2)
    check("search returns hits", len(res) >= 1, f"got {len(res)}")
    check("search top hit ranked correctly", res and "bird" in res[0].text.lower(), f"top={res[0].text if res else 'none'}")
    hres = vs.hybrid_search("cars motor", limit=2)
    check("hybrid_search works", len(hres) >= 1)
    vs.delete(ids)

    # ── 4. KnowledgeBase + chunking ──────────────────────────────
    section("4. KNOWLEDGE BASE (chunking, indexed wiki, combined query)")
    chunks = chunk_text("ABCDEFGHIJ" * 50, chunk_size=100, overlap=20)
    check("chunk_text produces overlapping chunks", len(chunks) > 1)
    check("chunks contain overlap", chunks[0][-20:] == chunks[1][:20])
    kb = KnowledgeBase(cfg)
    kb.add_page("Vector Search Notes", "Cosine similarity over normalized embeddings is fast and accurate.", "notes")
    kb.add_page("Graph Theory", "Cypher queries traverse labeled property graphs efficiently.", "notes")
    time.sleep(0.5)
    cq = kb.query("cosine similarity", wiki_limit=2, vector_limit=3)
    check("combined query returns hits", bool(cq))
    check("combined query top is relevant", cq and ("Vector" in cq[0].title or "vector" in cq[0].snippet.lower()))
    n = kb.reindex_all()
    check("reindex_all processes pages", n >= 4, f"reindexed {n}")

    # ── 5. Graph store ───────────────────────────────────────────
    section("5. GRAPH STORE (FalkorDB + entities + relations + GraphRAG)")
    gs = GraphStore(cfg)
    check("graph ping", gs.ping())
    gs.add_entity(Entity("Project", "RKE-Test", {"version": "0.1.1"}))
    gs.add_entity(Entity("API", "Qdrant"))
    gs.add_entity(Entity("API", "FalkorDB"))
    gs.add_relation(Relation("Project", "RKE-Test", "USES", "API", "Qdrant"))
    gs.add_relation(Relation("Project", "RKE-Test", "USES", "API", "FalkorDB"))
    stats = gs.stats()
    check("graph has nodes", stats.get("nodes", 0) >= 3, f"nodes={stats.get('nodes')}")
    check("graph has rels", stats.get("relationships", 0) >= 2, f"rels={stats.get('relationships')}")
    rows = gs.query("MATCH (p:Project)-[:USES]->(a:API) RETURN p.name, a.name")
    check("cypher query returns rows", len(rows) >= 2)
    rag = gs.graphrag_query("which APIs does RKE-Test use?")
    check("graphrag_query returns subgraph", "subgraph" in rag)

    # ── 6. RLM environment ───────────────────────────────────────
    section("6. RLM ENVIRONMENT (peek, grep, partition)")
    env: Environment = make_environment(cfg)
    peek_out = env.peek("vector-search-notes")
    check("peek returns content", "Cosine" in peek_out or "Vector" in peek_out)
    grep_hits = env.grep("Cypher", in_wiki=True, in_vectors=False, limit=5)
    check("grep finds matches", any("Cypher" in (h.get("text") or "") for h in grep_hits))
    parts = env.partition("knowledge graph patterns", k=2)
    check("partition returns sub-queries", len(parts) >= 1)
    check("env.trace is populated", len(env.trace) >= 3)

    # ── 7. RLM router (deterministic) ────────────────────────────
    section("7. RLM ROUTER (deterministic mode)")
    router = RLMRouter(config=cfg)
    res2 = router.complete("vector search cosine similarity")
    check("router returns answer", bool(res2.answer))
    check("router used deterministic mode", res2.used_llm is False)
    check("router answer mentions hits", "Vector Search Notes" in res2.answer or "Cosine" in res2.answer)

    # ── 8. Agent integration ─────────────────────────────────────
    section("8. AGENT INTEGRATION (gather_context, role formatting)")
    ctx = gather_context("cypher knowledge graph", config=cfg)
    check("gather_context returns hits", bool(ctx.hits))
    check("gather_context includes graph", isinstance(ctx.graph, dict))
    for agent in ("claude", "codex", "gemini", "generic"):
        out = format_context_for_agent(ctx, agent=agent)
        check(f"format for {agent}", "<context>" in out and "<evidence" in out)

    # ── 9. Ingestion: local files ────────────────────────────────
    section("9. INGESTION (local files via iter_files)")
    test_dir = Path("/tmp/rke_ingest_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    (test_dir / "doc1.md").write_text("# Doc 1\n\nProject docs about authentication patterns.")
    (test_dir / "doc2.md").write_text("# Doc 2\n\nDatabase tuning notes for postgres.")
    (test_dir / "code.py").write_text("def foo(): return 42\n")
    files = iter_files([str(test_dir)], ["**/*.md", "**/*.py"], [])
    check("iter_files finds 3", len(files) == 3, f"got {len(files)}")

    # ── 10. Ingestion: git_repos.list_repos ──────────────────────
    section("10. INGESTION (git repo discovery)")
    repos = list_repos(Path("/tmp/security-audit/RKEmemory"))
    check("list_repos finds RKEmemory", any("RKEmemory" in str(r) for r in repos))

    # ── 11. CLI smoke ────────────────────────────────────────────
    section("11. CLI (subprocess smoke for each command)")
    env_vars = {**os.environ}
    cli = ["python", "-m", "rke"]
    for args, expect_in_stdout in [
        (["version"], f"rke {__version__}"),
        (["--help"], "Recursive Knowledge Engine"),
        (["status"], "Qdrant"),
        (["wiki-list"], "OAuth"),
        (["lint"], None),
    ]:
        try:
            r = subprocess.run(
                cli + args, env=env_vars, capture_output=True, text=True, timeout=120,
            )
            ok = r.returncode == 0
            if expect_in_stdout:
                ok = ok and (expect_in_stdout in r.stdout or expect_in_stdout in r.stderr)
            check(f"CLI {args[0]}", ok, f"rc={r.returncode} stdout={r.stdout[:80]!r}")
        except Exception as exc:
            check(f"CLI {args[0]}", False, str(exc))

    # ── Summary ──────────────────────────────────────────────────
    section("RESULT")
    print(f"  passed: {len(PASSED)}")
    print(f"  failed: {len(FAILED)}")
    if FAILED:
        for name, det in FAILED:
            print(f"    [FAIL] {name}: {det}")
        return 1
    print("\n  ALL FEATURES WORKING ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
