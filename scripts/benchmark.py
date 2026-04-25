"""RKE benchmark — measures the dimensions that matter for a memory system.

Reports:
  • embedding throughput (docs/sec)
  • ingestion throughput (pages/sec, end-to-end)
  • query latency p50/p95/p99 (vector, hybrid, combined wiki+vector+graph)
  • recall@1 / recall@5 over a synthetic ground-truth set
  • memory footprint (RSS) before and after a 200-doc corpus
"""

from __future__ import annotations

import os
import random
import shutil
import statistics
import time
from pathlib import Path

import psutil

from rke.config import load_config
from rke.graph_store import Entity, GraphStore, Relation
from rke.vector_store import VectorStore
from rke.wiki.knowledge_base import KnowledgeBase

random.seed(42)


# Synthetic corpus: 10 topic clusters × 20 docs each = 200 docs.
TOPICS = {
    "auth": "OAuth refresh tokens are rotated when users switch accounts. Authentication uses bearer tokens.",
    "db":   "Postgres tuning involves work_mem and shared_buffers. Index B-trees, vacuum analyze regularly.",
    "vec":  "Cosine similarity over normalized embeddings powers semantic search. ANN indexes use HNSW.",
    "graph":"Cypher queries traverse labeled property graphs. Knowledge graphs encode entity relationships.",
    "k8s":  "Kubernetes deployments orchestrate containers. Pods are scheduled by the kube-scheduler.",
    "ml":   "Transformer attention computes weighted token representations. Models train on GPU clusters.",
    "web":  "React components manage UI state. Server-side rendering improves first-paint latency.",
    "api":  "REST endpoints expose JSON. GraphQL queries return only requested fields. gRPC is binary.",
    "sec":  "TLS certificates verify identity. Symmetric keys encrypt session payloads. RBAC limits access.",
    "obs":  "Distributed tracing follows requests across services. Metrics use Prometheus, logs go to Loki.",
}


def synth_corpus(n_per_topic: int = 20) -> list[tuple[str, str, str]]:
    """Returns list of (topic, title, body)."""
    out = []
    for topic, base in TOPICS.items():
        for i in range(n_per_topic):
            sentences = [base]
            # add some lexical variation
            sentences.append(f"Variant {i}: {base.lower()}")
            sentences.append(f"Detail {i}: noted by team in sprint {i}.")
            out.append((topic, f"{topic} doc {i}", " ".join(sentences)))
    random.shuffle(out)
    return out


# Queries with known correct topic.
QUERIES = [
    ("auth", "how do refresh tokens work when switching users"),
    ("db", "postgres performance tuning shared buffers"),
    ("vec", "cosine similarity semantic vector search"),
    ("graph", "cypher property graph traversal"),
    ("k8s", "kubernetes pod scheduling"),
    ("ml", "transformer attention mechanism"),
    ("web", "react component state management"),
    ("api", "graphql versus rest endpoints"),
    ("sec", "tls certificate verification"),
    ("obs", "prometheus metrics distributed tracing"),
]


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = int(len(s) * pct / 100)
    return s[min(k, len(s) - 1)]


def bench(label: str, fn, n: int = 30) -> dict[str, float]:
    """Time `fn` n times. Discard first call (warmup) to exclude one-time cost."""
    fn()  # warmup
    timings = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1000)  # ms
    return {
        "label": label,
        "n": n,
        "p50_ms": percentile(timings, 50),
        "p95_ms": percentile(timings, 95),
        "p99_ms": percentile(timings, 99),
        "mean_ms": statistics.mean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def main() -> int:
    cfg = load_config()
    proc = psutil.Process(os.getpid())
    print(f"=== RKE benchmark — {cfg.embedding['model']} ({cfg.embedding['dimensions']}d) on {cfg.embedding['device']}")
    print(f"qdrant={cfg.qdrant['host']}:{cfg.qdrant['port']}  falkordb={cfg.falkordb['host']}:{cfg.falkordb['port']}")

    # Clean slate
    if cfg.wiki_path.exists():
        shutil.rmtree(cfg.wiki_path)
    vs0 = VectorStore(cfg)
    try:
        vs0.client.delete_collection(cfg.qdrant["collection"])
    except Exception:
        pass

    rss_idle_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"\nRSS at idle:  {rss_idle_mb:>7.1f} MB")

    # ── 1. Embedding throughput ─────────────────────────────────
    print("\n--- 1. Embedding throughput (raw encode) ---")
    vs = VectorStore(cfg)
    _ = vs.embedder  # force load
    rss_loaded_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"RSS after model load: {rss_loaded_mb:>7.1f} MB")
    sample = [TOPICS["auth"]] * 100
    t0 = time.perf_counter()
    vs.embed(sample)
    embed_dt = time.perf_counter() - t0
    embed_throughput = len(sample) / embed_dt
    print(f"100 docs in {embed_dt*1000:.1f} ms → {embed_throughput:.1f} docs/sec")

    # ── 2. Ingestion throughput (end-to-end) ────────────────────
    print("\n--- 2. Ingestion throughput (wiki + vector store) ---")
    kb = KnowledgeBase(cfg)
    corpus = synth_corpus(n_per_topic=20)  # 200 docs
    t0 = time.perf_counter()
    for topic, title, body in corpus:
        kb.add_page(title=title, body=body, category=f"bench/{topic}", tags=[topic])
    ingest_dt = time.perf_counter() - t0
    ingest_throughput = len(corpus) / ingest_dt
    print(f"{len(corpus)} pages in {ingest_dt:.2f} s → {ingest_throughput:.1f} pages/sec")
    rss_indexed_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"RSS after 200-doc index: {rss_indexed_mb:>7.1f} MB")

    time.sleep(0.5)  # let qdrant settle

    # Seed the graph for combined-query benchmarks
    gs = GraphStore(cfg)
    gs.connect()
    for topic in TOPICS:
        gs.add_entity(Entity("Topic", topic))
        gs.add_entity(Entity("Project", "RKE"))
        gs.add_relation(Relation("Project", "RKE", "COVERS", "Topic", topic))

    # ── 3. Query latency ────────────────────────────────────────
    print("\n--- 3. Query latency (n=30, warmup discarded) ---")
    sample_q = "how do refresh tokens work when switching users"
    vec_stats = bench("vector search (k=5)", lambda: vs.search(sample_q, limit=5))
    hyb_stats = bench("hybrid search (k=5)", lambda: vs.hybrid_search(sample_q, limit=5))
    com_stats = bench("combined kb.query (k=5+5)", lambda: kb.query(sample_q, wiki_limit=5, vector_limit=5))
    for s in (vec_stats, hyb_stats, com_stats):
        print(f"  {s['label']:<32} p50={s['p50_ms']:>6.1f}ms  p95={s['p95_ms']:>6.1f}ms  p99={s['p99_ms']:>6.1f}ms  mean={s['mean_ms']:>6.1f}ms")

    # ── 4. Recall@K ─────────────────────────────────────────────
    print("\n--- 4. Recall@K against synthetic ground truth (10 queries × 10 topics) ---")
    r1, r5 = 0, 0
    for topic, q in QUERIES:
        hits = vs.search(q, limit=5)
        if hits:
            tags_top1 = (hits[0].metadata.get("tags") or [None])[0]
            if tags_top1 == topic:
                r1 += 1
            for h in hits:
                if (h.metadata.get("tags") or [None])[0] == topic:
                    r5 += 1
                    break
    print(f"  Recall@1: {r1}/{len(QUERIES)} = {100*r1/len(QUERIES):.0f}%")
    print(f"  Recall@5: {r5}/{len(QUERIES)} = {100*r5/len(QUERIES):.0f}%")

    # ── 5. Graph + GraphRAG latency ─────────────────────────────
    print("\n--- 5. Graph latency ---")
    cypher_stats = bench("cypher MATCH 1-hop", lambda: gs.query("MATCH (p:Project)-[:COVERS]->(t:Topic) RETURN t.name LIMIT 10"))
    rag_stats = bench("graphrag_query", lambda: gs.graphrag_query("which topics does RKE cover?"))
    for s in (cypher_stats, rag_stats):
        print(f"  {s['label']:<32} p50={s['p50_ms']:>6.1f}ms  p95={s['p95_ms']:>6.1f}ms  p99={s['p99_ms']:>6.1f}ms")

    print("\n=== Summary ===")
    print(f"  Embedding throughput:  {embed_throughput:>7.1f} docs/sec")
    print(f"  Ingestion throughput:  {ingest_throughput:>7.1f} pages/sec (200-doc corpus, end-to-end)")
    print(f"  Vector query p50/p95:  {vec_stats['p50_ms']:.1f} / {vec_stats['p95_ms']:.1f} ms")
    print(f"  Combined query p50/p95:{com_stats['p50_ms']:.1f} / {com_stats['p95_ms']:.1f} ms")
    print(f"  Recall@1 / @5:         {100*r1/len(QUERIES):.0f}% / {100*r5/len(QUERIES):.0f}%")
    print(f"  RSS:                   idle {rss_idle_mb:.0f} → loaded {rss_loaded_mb:.0f} → indexed {rss_indexed_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
