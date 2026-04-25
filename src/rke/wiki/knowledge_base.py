"""KnowledgeBase — bridges WikiManager + VectorStore.

Adds a wiki page and indexes it into the vector store at the same time, so
semantic search and keyword wiki lookups stay consistent.

If `llama-index` is installed, a `LlamaIndexBackend` is also available; the
default backend uses our own VectorStore directly to keep dependencies light.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, load_config
from ..vector_store import SearchResult, VectorStore
from .manager import WikiManager, WikiPage

log = logging.getLogger(__name__)


@dataclass
class CombinedHit:
    source: str  # "wiki" or "vector"
    score: float
    title: str
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 128) -> list[str]:
    """Character-level chunker. Cheap and language-agnostic."""
    if chunk_size <= 0:
        return [text]
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
    return chunks


class KnowledgeBase:
    def __init__(
        self,
        config: Config | None = None,
        wiki: WikiManager | None = None,
        vectors: VectorStore | None = None,
    ) -> None:
        self.config = config or load_config()
        self.wiki = wiki or WikiManager(self.config)
        self.vectors = vectors or VectorStore(self.config)

    # ── Ingest ───────────────────────────────────────────────────
    def add_page(
        self,
        title: str,
        body: str,
        category: str = "general",
        tags: list[str] | None = None,
    ) -> WikiPage:
        page = self.wiki.create_page(title, body, category=category, tags=tags)
        self._index_page(page)
        return page

    def _index_page(self, page: WikiPage) -> None:
        cs = int(self.config.ingestion.get("chunk_size", 1024))
        ov = int(self.config.ingestion.get("chunk_overlap", 128))
        chunks = chunk_text(page.body, chunk_size=cs, overlap=ov)
        if not chunks:
            return
        metas = [
            {
                "slug": page.slug,
                "title": page.title,
                "category": page.category,
                "tags": page.tags,
                "chunk_index": i,
                "source": "wiki",
            }
            for i in range(len(chunks))
        ]
        # Deterministic IDs scoped by (category, slug) so same-slug-different-
        # category pages do not collide in the vector index.
        prefix = f"wiki:{page.category}:{page.slug}"
        ids = [f"{prefix}:{i}" for i in range(len(chunks))]
        # Convert to UUIDs since Qdrant requires UUID/int point IDs.
        import uuid as _uuid
        uuid_ids = [str(_uuid.uuid5(_uuid.NAMESPACE_URL, i)) for i in ids]

        # Before upsert, evict any STALE chunks left over from a previous
        # longer revision of this page. We delete every existing point
        # whose payload matches this (category, slug) but whose
        # chunk_index is beyond the new chunk count.
        try:
            self.vectors.delete_by_filter(
                {"category": page.category, "slug": page.slug},
                min_chunk_index=len(chunks),
            )
        except Exception as exc:
            log.warning("stale-chunk cleanup skipped: %s", exc)

        self.vectors.upsert(chunks, metadatas=metas, ids=uuid_ids)

    def reindex_all(self) -> int:
        """Rebuild the vector index from the wiki. Returns # pages indexed."""
        n = 0
        for page in self.wiki.list_pages():
            self._index_page(page)
            n += 1
        return n

    # ── Combined query ───────────────────────────────────────────
    def query(
        self,
        question: str,
        wiki_limit: int = 3,
        vector_limit: int = 5,
    ) -> list[CombinedHit]:
        """Run wiki keyword + vector semantic search, merge ranked."""
        hits: list[CombinedHit] = []
        for page in self.wiki.query_wiki(question, limit=wiki_limit):
            hits.append(CombinedHit(
                source="wiki",
                score=1.0,
                title=page.title,
                snippet=page.body[:300],
                metadata={"slug": page.slug, "category": page.category, "tags": page.tags},
            ))
        try:
            for r in self.vectors.hybrid_search(question, limit=vector_limit):
                hits.append(CombinedHit(
                    source="vector",
                    score=r.score,
                    title=r.metadata.get("title", "(chunk)"),
                    snippet=r.text[:300],
                    metadata=r.metadata,
                ))
        except Exception as exc:  # vector store may not be reachable
            log.warning("vector search unavailable: %s", exc)
        # Stable, score-aware merge — wiki hits keep their score=1.0 prior.
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    @staticmethod
    def search_results_to_hits(results: list[SearchResult]) -> list[CombinedHit]:
        return [
            CombinedHit(
                source="vector",
                score=r.score,
                title=r.metadata.get("title", "(chunk)"),
                snippet=r.text[:300],
                metadata=r.metadata,
            )
            for r in results
        ]
