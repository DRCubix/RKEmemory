"""Qdrant + sentence-transformers vector store.

Wraps Qdrant for semantic and hybrid search. Embeddings are produced locally
via sentence-transformers (default: BGE-M3, 1024-dim).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from .config import Config, load_config

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore:
    """Thin Qdrant wrapper. Lazy-loads the embedding model on first use."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self._client = None
        self._embedder = None
        self._dim: int | None = None
        self._collection = self.config.qdrant.get("collection", "rke_main")

    # ── Connection ───────────────────────────────────────────────
    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            qcfg = self.config.qdrant
            self._client = QdrantClient(
                host=qcfg.get("host", "localhost"),
                port=qcfg.get("port", 6333),
                grpc_port=qcfg.get("grpc_port", 6334),
                api_key=qcfg.get("api_key") or None,
                prefer_grpc=False,
            )
        return self._client

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            ecfg = self.config.embedding
            model_name = ecfg.get("model", "BAAI/bge-m3")
            device = ecfg.get("device", "cpu")
            log.info("loading embedding model %s on %s", model_name, device)
            self._embedder = SentenceTransformer(model_name, device=device)
            # sentence-transformers >=3.0 renamed this method.
            getter = getattr(self._embedder, "get_embedding_dimension", None) \
                or self._embedder.get_sentence_embedding_dimension
            self._dim = getter()
        return self._embedder

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Trigger lazy load OR fall back to config value.
            cfg_dim = self.config.embedding.get("dimensions")
            if cfg_dim:
                self._dim = int(cfg_dim)
            else:
                _ = self.embedder  # forces population
        return int(self._dim or 1024)

    # ── Collection management ────────────────────────────────────
    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        from qdrant_client.http.models import Distance, VectorParams

        existing = {c.name for c in self.client.get_collections().collections}
        if self._collection in existing:
            return
        log.info("creating qdrant collection %s (dim=%d)", self._collection, self.dim)
        self.client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def collection_info(self) -> dict[str, Any]:
        try:
            info = self.client.get_collection(self._collection)
            out: dict[str, Any] = {
                "name": self._collection,
                "points_count": getattr(info, "points_count", None),
                "status": str(getattr(info, "status", "")),
            }
            # vectors_count was removed in newer qdrant-client; tolerate both
            vc = getattr(info, "vectors_count", None)
            if vc is not None:
                out["vectors_count"] = vc
            return out
        except Exception as exc:
            return {"name": self._collection, "error": str(exc)}

    # ── Embedding ────────────────────────────────────────────────
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = int(self.config.embedding.get("batch_size", 32))
        vectors = self.embedder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    # ── Upsert ───────────────────────────────────────────────────
    def upsert(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Embed and upsert. Returns the IDs assigned."""
        from qdrant_client.http.models import PointStruct

        if not texts:
            return []
        self.ensure_collection()
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        vectors = self.embed(texts)

        points = [
            PointStruct(
                id=pid,
                vector=vec,
                payload={"text": txt, **md},
            )
            for pid, vec, txt, md in zip(ids, vectors, texts, metadatas, strict=True)
        ]
        self.client.upsert(collection_name=self._collection, points=points)
        return ids

    # ── Search ───────────────────────────────────────────────────
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Pure semantic search."""
        self.ensure_collection()
        qvec = self.embed([query])[0]
        # qdrant-client 1.10+ deprecated .search() in favor of .query_points().
        # Use query_points() if available, fall back to legacy search().
        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(
                collection_name=self._collection,
                query=qvec,
                limit=limit,
                with_payload=True,
            )
            points = getattr(resp, "points", resp)
        else:
            points = self.client.search(  # type: ignore[attr-defined]
                collection_name=self._collection,
                query_vector=qvec,
                limit=limit,
            )
        return [self._to_result(h) for h in points]

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        keyword_boost: float = 0.2,
    ) -> list[SearchResult]:
        """Semantic search with simple keyword overlap re-ranking.

        FalkorDB-grade hybrid (BM25 + dense fusion) is on the roadmap; this
        is a deterministic, dependency-free approximation that boosts hits
        whose payload text shares tokens with the query.
        """
        results = self.search(query, limit=max(limit * 2, limit + 5))
        if not results:
            return []
        q_tokens = {t.lower() for t in query.split() if len(t) > 2}
        rescored: list[SearchResult] = []
        for r in results:
            text_tokens = {t.lower() for t in r.text.split() if len(t) > 2}
            overlap = len(q_tokens & text_tokens)
            denom = max(len(q_tokens), 1)
            bonus = keyword_boost * (overlap / denom)
            rescored.append(SearchResult(
                id=r.id,
                score=min(r.score + bonus, 1.0),
                text=r.text,
                metadata=r.metadata,
            ))
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:limit]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.client.delete(collection_name=self._collection, points_selector=ids)

    def delete_by_filter(
        self,
        match: dict[str, Any],
        *,
        min_chunk_index: int | None = None,
    ) -> int:
        """Delete every point whose payload exactly matches each key/value in
        ``match``. If ``min_chunk_index`` is given, only points with
        ``payload.chunk_index >= min_chunk_index`` are removed (the rest are
        about to be overwritten by a fresh upsert).

        Used by KnowledgeBase to evict stale chunks left over from a longer
        previous revision of a page.
        """
        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
            Range,
        )

        try:
            self.ensure_collection()
        except Exception:
            return 0

        must = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in match.items()
        ]
        if min_chunk_index is not None:
            must.append(FieldCondition(
                key="chunk_index",
                range=Range(gte=float(min_chunk_index)),
            ))
        flt = Filter(must=must)

        try:
            self.client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=flt),
            )
            return 1
        except Exception as exc:
            log.warning("delete_by_filter failed: %s", exc)
            return 0

    # ── Helpers ──────────────────────────────────────────────────
    @staticmethod
    def _to_result(hit) -> SearchResult:
        payload = dict(hit.payload or {})
        text = payload.pop("text", "")
        return SearchResult(
            id=str(hit.id),
            score=float(hit.score),
            text=text,
            metadata=payload,
        )
