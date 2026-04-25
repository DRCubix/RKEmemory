"""Whoosh-backed inverted-index accelerator for wiki keyword search.

Drop-in accelerator for :func:`rke.wiki.manager.WikiManager.query_wiki`.
Maintains a BM25F-scored index of WikiPages on disk and wires into the
WikiManager hook system so add/delete stay in sync.

Usage::

    from rke.wiki.manager import WikiManager
    from rke.wiki.search_index import WhooshIndex

    wm = WikiManager()
    idx = WhooshIndex.attach(wm, index_dir=wm.root.parent / ".wiki_index")
    wm.query_wiki("oauth tokens")  # now index-accelerated
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from whoosh import index as whoosh_index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, KEYWORD, TEXT, Schema
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.scoring import BM25F

from . import manager as wiki_manager
from .manager import WikiManager, WikiPage

log = logging.getLogger(__name__)


def _build_schema() -> Schema:
    """Schema with title/tags boosted so matches there outrank body-only hits."""
    stem = StemmingAnalyzer()
    return Schema(
        slug=ID(stored=True, unique=True),
        title=TEXT(stored=True, analyzer=stem, field_boost=5.0),
        tags=KEYWORD(stored=True, commas=True, lowercase=True, scorable=True,
                     field_boost=2.0),
        body=TEXT(stored=False, analyzer=stem),
    )


class WhooshIndex:
    """Disk-backed inverted index for WikiPage objects."""

    _attached: WhooshIndex | None = None

    def __init__(self, index_dir: Path | str) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if whoosh_index.exists_in(str(self.index_dir)):
            self._ix = whoosh_index.open_dir(str(self.index_dir))
        else:
            self._ix = whoosh_index.create_in(str(self.index_dir), _build_schema())

    # ── CRUD ─────────────────────────────────────────────────────
    def add(self, page: WikiPage) -> None:
        """Index or replace a page (slug is the unique key)."""
        writer = self._ix.writer()
        try:
            writer.update_document(
                slug=str(page.slug),
                title=str(page.title or ""),
                tags=",".join(str(t) for t in (page.tags or [])),
                body=str(page.body or ""),
            )
            writer.commit()
        except Exception:
            writer.cancel()
            raise

    def remove(self, slug: str) -> None:
        """Delete the document with the given slug, if present."""
        writer = self._ix.writer()
        try:
            writer.delete_by_term("slug", str(slug))
            writer.commit()
        except Exception:
            writer.cancel()
            raise

    # ── Query ────────────────────────────────────────────────────
    def search(self, query: str, limit: int = 5) -> list[str]:
        """Return ranked slugs (best first) for the given query string."""
        q = (query or "").strip()
        if not q:
            return []
        parser = MultifieldParser(
            ["title", "tags", "body"], schema=self._ix.schema, group=OrGroup
        )
        try:
            parsed = parser.parse(q)
        except Exception as exc:
            log.warning("whoosh parse failed for %r: %s", q, exc)
            return []
        with self._ix.searcher(weighting=BM25F()) as searcher:
            results = searcher.search(parsed, limit=limit)
            return [hit["slug"] for hit in results if "slug" in hit]

    # ── Rebuild ──────────────────────────────────────────────────
    def rebuild(self, pages: Iterable[WikiPage]) -> int:
        """Wipe the index and repopulate from `pages`. Returns count added."""
        self._ix.close()
        self._ix = whoosh_index.create_in(str(self.index_dir), _build_schema())
        writer = self._ix.writer()
        count = 0
        try:
            for page in pages:
                writer.add_document(
                    slug=str(page.slug),
                    title=str(page.title or ""),
                    tags=",".join(str(t) for t in (page.tags or [])),
                    body=str(page.body or ""),
                )
                count += 1
            writer.commit()
        except Exception:
            writer.cancel()
            raise
        return count

    # ── Attach / detach ──────────────────────────────────────────
    @classmethod
    def attach(cls, wm: WikiManager, index_dir: Path | str) -> WhooshIndex:
        """Create a WhooshIndex, wire hooks, and do an idempotent initial rebuild.

        Subsequent create/delete calls on `wm` will keep the index in sync and
        `wm.query_wiki()` will route through the BM25F backend.
        """
        idx = cls(index_dir)
        idx.rebuild(wm.list_pages())

        wiki_manager.register_post_create(idx.add)
        wiki_manager.register_post_delete(idx.remove)
        wiki_manager.set_query_backend(idx.search)

        cls._attached = idx
        return idx

    @classmethod
    def detach(cls) -> None:
        """Unwire ONLY this index's hooks. Other plugins (extractor,
        lifecycle, ...) keep their registrations intact."""
        if cls._attached is None:
            return
        idx = cls._attached
        try:
            wiki_manager._post_create_hooks.remove(idx.add)
        except (ValueError, AttributeError):
            pass
        try:
            wiki_manager._post_delete_hooks.remove(idx.remove)
        except (ValueError, AttributeError):
            pass
        if wiki_manager._query_backend is idx.search:
            wiki_manager.set_query_backend(None)
        try:
            idx._ix.close()
        except Exception:
            pass
        cls._attached = None
