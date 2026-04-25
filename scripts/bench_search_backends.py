"""Head-to-head benchmark: Whoosh vs Tantivy backends.

Measures index build time and query latency p50/p95/p99 on a 200-doc
synthetic corpus. Both backends present the same `search(query, limit)`
API so this is a true apples-to-apples comparison.

Run: PYTHONPATH=src python scripts/bench_search_backends.py
"""

from __future__ import annotations

import math
import os
import shutil
import statistics
import sys
import tempfile
import time
import uuid as _uuid_mod
from pathlib import Path

# No services needed for this micro-benchmark.
os.environ.setdefault("RKE_WIKI_PATH", f"/tmp/rke_search_bench_{_uuid_mod.uuid4().hex[:6]}")

from rke.wiki.manager import WikiPage  # noqa: E402

try:
    from rke.wiki.search_index import WhooshIndex  # noqa: E402
except ImportError as exc:
    print(
        f"ERROR: missing whoosh ({exc}). Install with:\n"
        '  pip install -e ".[search]"',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

try:
    from rke.wiki.tantivy_index import TantivyIndex  # noqa: E402
except ImportError as exc:
    print(
        f"ERROR: missing tantivy ({exc}). Install with:\n"
        '  pip install -e ".[search-tantivy]"',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

TOPICS = {
    "auth":  "OAuth refresh tokens are rotated when users switch accounts. Bearer tokens authenticate.",
    "db":    "Postgres tuning involves work_mem and shared_buffers. Index B-trees, vacuum analyze.",
    "vec":   "Cosine similarity over normalized embeddings. ANN indexes use HNSW for fast lookup.",
    "graph": "Cypher queries traverse labeled property graphs. Knowledge graphs encode entities.",
    "k8s":   "Kubernetes deployments orchestrate containers. Pods are scheduled by kube-scheduler.",
    "ml":    "Transformer attention computes weighted token representations. GPU clusters train.",
    "web":   "React components manage UI state. Server-side rendering improves first-paint latency.",
    "api":   "REST endpoints expose JSON. GraphQL queries return only requested fields.",
    "sec":   "TLS certificates verify identity. Symmetric keys encrypt session payloads. RBAC.",
    "obs":   "Distributed tracing follows requests across services. Prometheus metrics, Loki logs.",
}

QUERIES = [
    "oauth refresh tokens",
    "postgres tuning shared buffers",
    "cosine similarity hnsw",
    "cypher graph traversal",
    "kubernetes pod scheduling",
    "transformer attention",
    "react state management",
    "graphql json endpoints",
    "tls certificate verification",
    "prometheus tracing logs",
]


def make_corpus(n_per_topic: int = 20) -> list[WikiPage]:
    out = []
    for topic, base in TOPICS.items():
        for i in range(n_per_topic):
            body = f"{base} Variant {i}: {base.lower()} Detail {i}: noted in sprint {i}."
            out.append(WikiPage(
                title=f"{topic} doc {i}",
                body=body,
                category=f"bench/{topic}",
                tags=[topic],
                slug=f"{topic}-doc-{i}",
            ))
    return out


def percentile(data: list[float], pct: float) -> float:
    """Nearest-rank percentile per the standard convention. For 300 samples
    p95 returns the 285th sorted value (index 284), not the 286th."""
    if not data:
        return 0.0
    s = sorted(data)
    k = max(0, math.ceil(len(s) * pct / 100) - 1)
    return s[min(k, len(s) - 1)]


def bench(name: str, idx, queries: list[str], limit: int = 5, n_iter: int = 30):
    # Warmup.
    for q in queries:
        idx.search(q, limit=limit)
    timings = []
    for _ in range(n_iter):
        for q in queries:
            t0 = time.perf_counter()
            idx.search(q, limit=limit)
            timings.append((time.perf_counter() - t0) * 1000)
    print(f"  {name:<10} p50={percentile(timings, 50):>5.2f} ms  "
          f"p95={percentile(timings, 95):>5.2f} ms  "
          f"p99={percentile(timings, 99):>5.2f} ms  "
          f"mean={statistics.mean(timings):>5.2f} ms  "
          f"({len(timings)} samples)")
    return timings


def main() -> int:
    corpus = make_corpus(n_per_topic=20)
    print(f"corpus: {len(corpus)} pages, {len(QUERIES)} unique queries")

    # ── Build phase ────────────────────────────────────────────
    print("\n--- Index build (200 pages from cold) ---")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # Whoosh
        widx = WhooshIndex(d / "whoosh")
        t0 = time.perf_counter()
        widx.rebuild(corpus)
        whoosh_build = (time.perf_counter() - t0) * 1000
        # Tantivy
        tidx = TantivyIndex(d / "tantivy")
        t0 = time.perf_counter()
        tidx.rebuild(corpus)
        tantivy_build = (time.perf_counter() - t0) * 1000
        print(f"  whoosh   : {whoosh_build:>7.1f} ms")
        print(f"  tantivy  : {tantivy_build:>7.1f} ms  ({whoosh_build / tantivy_build:.1f}× speedup)")

        # ── Query latency ──────────────────────────────────────
        print("\n--- Query latency (10 queries × 30 iterations = 300 samples) ---")
        whoosh_t = bench("whoosh", widx, QUERIES)
        tantivy_t = bench("tantivy", tidx, QUERIES)

        print("\n--- Speedup ---")
        print(f"  p50:  {percentile(whoosh_t, 50) / percentile(tantivy_t, 50):.1f}×")
        print(f"  p95:  {percentile(whoosh_t, 95) / percentile(tantivy_t, 95):.1f}×")
        print(f"  mean: {statistics.mean(whoosh_t) / statistics.mean(tantivy_t):.1f}×")

        # Cleanup.
        shutil.rmtree(d / "whoosh", ignore_errors=True)
        shutil.rmtree(d / "tantivy", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
