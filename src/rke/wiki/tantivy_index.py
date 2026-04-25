"""Tantivy-backed inverted-index accelerator for wiki keyword search.

Drop-in alternative to :class:`rke.wiki.search_index.WhooshIndex` with
the same public API but backed by Tantivy (Rust). Measured ~17–21×
faster than Whoosh on query latency for a 200-doc corpus
(see ``scripts/bench_search_backends.py`` for reproducible numbers).
Install with ``pip install -e ".[search-tantivy]"``.

The schema, composite ``(category, slug)`` key, hook integration, and
tenant scoping all mirror :class:`WhooshIndex` exactly so callers can
swap the implementation without changing application code.

Usage::

    from rke.wiki.manager import WikiManager
    from rke.wiki.tantivy_index import TantivyIndex

    wm = WikiManager()
    idx = TantivyIndex.attach(wm, index_dir=wm.root.parent / ".wiki_idx")
    wm.query_wiki("oauth tokens")  # BM25F-ranked
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import tantivy

from . import manager as wiki_manager
from .manager import WikiManager, WikiPage

log = logging.getLogger(__name__)

# Tantivy field boost multipliers passed to parse_query(field_boosts=...) —
# title 5x and tags 2x mirror the Whoosh schema.
_FIELD_BOOSTS = {"title": 5.0, "tags": 2.0, "body": 1.0}
_DEFAULT_FIELDS = ["title", "tags", "body"]


def _path(category: str, slug: str) -> str:
    """Composite '(category, slug)' identity. Same wire format as WhooshIndex
    so :meth:`WikiManager.query_wiki` can split it identically."""
    return f"{category}/{slug}"


# Bump whenever _build_schema() changes. Persisted next to the index so we
# detect (and refuse to open) stale-but-valid Tantivy directories built by
# a previous schema variant.
_SCHEMA_VERSION = "v2-en_stem-raw_tags"


def _build_schema() -> tantivy.Schema:
    """Schema mirroring search_index.WhooshIndex: composite path is the
    unique key; slug + category are individually queryable; title/body
    use English stemming (Whoosh StemmingAnalyzer parity); tags use the
    `raw` tokenizer with one term per tag, mirroring Whoosh's
    KEYWORD(commas=True, lowercase=True) semantics — values like
    'thread-archive' stay as a single token instead of splitting on '-'."""
    sb = tantivy.SchemaBuilder()
    # ID-like fields use the "raw" tokenizer so 'general/oauth-tokens' stays
    # as a single token instead of being split on '/' or '-'.
    sb.add_text_field("path", stored=True, tokenizer_name="raw")
    sb.add_text_field("slug", stored=True, tokenizer_name="raw")
    sb.add_text_field("category", stored=True, tokenizer_name="raw")
    # Stemmed text fields match Whoosh's StemmingAnalyzer semantics.
    sb.add_text_field("title", stored=True, tokenizer_name="en_stem")
    # Per-tag indexing under the raw tokenizer: each tag is added as its
    # own value via repeated doc.add_text("tags", tag) and stays whole.
    sb.add_text_field("tags", stored=True, tokenizer_name="raw")
    sb.add_text_field("body", stored=False, tokenizer_name="en_stem")
    return sb.build()


def _add_tags(doc: tantivy.Document, tags) -> None:
    """Index each tag as its own raw-tokenized value (one term per tag).
    Mirrors Whoosh KEYWORD(commas=True, lowercase=True) precisely:
    'thread-archive' stays whole, lowercase normalisation applied."""
    for tag in (tags or []):
        s = str(tag).strip().lower()
        if s:
            doc.add_text("tags", s)


class TantivyIndex:
    """Disk-backed Tantivy inverted index for WikiPage objects."""

    _attached: TantivyIndex | None = None

    def __init__(self, index_dir: Path | str) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._schema = _build_schema()
        # Tantivy persists its metadata as `meta.json` in the index dir.
        # We use that AND a per-RKE schema-version sentinel: an existing
        # tantivy index built with the old schema would still pass the
        # meta.json check, but its tags would be tokenised with the
        # default analyser instead of `raw`, silently degrading recall.
        # Refusing to open a mismatched-version directory makes the
        # incompatibility loud rather than silent.
        meta_path = self.index_dir / "meta.json"
        version_path = self.index_dir / ".rke_schema_version"
        if meta_path.exists():
            existing_version = (
                version_path.read_text().strip()
                if version_path.exists() else ""
            )
            if existing_version != _SCHEMA_VERSION:
                raise RuntimeError(
                    f"Tantivy index at {self.index_dir} was built with an "
                    f"incompatible schema (got {existing_version!r}, "
                    f"expected {_SCHEMA_VERSION!r}). "
                    "Delete the directory and rebuild from the wiki."
                )
            self._ix = tantivy.Index.open(str(self.index_dir))
        else:
            self._ix = tantivy.Index(self._schema, path=str(self.index_dir))
            version_path.write_text(_SCHEMA_VERSION)
        # Optional tenant root — set by attach() to filter foreign pages.
        self._tenant_root: Path | None = None

    # ── Tenant guard ─────────────────────────────────────────────
    def _belongs_to_tenant(self, page: WikiPage) -> bool:
        if self._tenant_root is None:
            return True
        if getattr(page, "path", None) is None:
            return False
        try:
            return page.path.resolve().is_relative_to(self._tenant_root)
        except Exception:
            return False

    # ── CRUD ─────────────────────────────────────────────────────
    def add(self, page: WikiPage) -> None:
        """Index or replace a page. Identity is (category, slug)."""
        if not self._belongs_to_tenant(page):
            return
        writer = self._ix.writer()
        try:
            # Delete existing doc with the same composite path so update
            # semantics match WhooshIndex.update_document.
            writer.delete_documents("path", _path(page.category, page.slug))
            doc = tantivy.Document()
            doc.add_text("path", _path(page.category, page.slug))
            doc.add_text("slug", str(page.slug))
            doc.add_text("category", str(page.category))
            doc.add_text("title", str(page.title or ""))
            _add_tags(doc, page.tags)
            doc.add_text("body", str(page.body or ""))
            writer.add_document(doc)
            writer.commit()
        except Exception:
            writer.rollback()
            raise
        self._ix.reload()

    def remove(self, page_or_slug: WikiPage | str) -> None:
        """Delete the document for ``page_or_slug``.

        Preferred: pass a :class:`WikiPage` so we target the exact
        ``(category, slug)``. Bare slug fallback wipes every category."""
        if isinstance(page_or_slug, WikiPage) and not self._belongs_to_tenant(page_or_slug):
            return
        writer = self._ix.writer()
        try:
            if isinstance(page_or_slug, WikiPage):
                writer.delete_documents(
                    "path", _path(page_or_slug.category, page_or_slug.slug),
                )
            else:
                writer.delete_documents("slug", str(page_or_slug))
            writer.commit()
        except Exception:
            writer.rollback()
            raise
        self._ix.reload()

    # ── Query ────────────────────────────────────────────────────
    def search(self, query: str, limit: int = 5) -> list[str]:
        """Return ranked composite identifiers ('category/slug')."""
        q = (query or "").strip()
        if not q:
            return []
        try:
            parsed = self._ix.parse_query(
                q, _DEFAULT_FIELDS, field_boosts=_FIELD_BOOSTS,
            )
        except Exception as exc:
            log.warning("tantivy parse failed for %r: %s", q, exc)
            return []
        searcher = self._ix.searcher()
        try:
            results = searcher.search(parsed, limit)
        except Exception as exc:
            log.warning("tantivy search failed for %r: %s", q, exc)
            return []
        out: list[str] = []
        for _score, addr in results.hits:
            try:
                doc = searcher.doc(addr).to_dict()
                paths = doc.get("path") or []
                if paths:
                    out.append(paths[0])
            except Exception:
                continue
        return out

    # ── Rebuild ──────────────────────────────────────────────────
    def rebuild(self, pages: Iterable[WikiPage]) -> int:
        """Wipe and repopulate from `pages`. Returns count added."""
        writer = self._ix.writer()
        try:
            writer.delete_all_documents()
            count = 0
            for page in pages:
                doc = tantivy.Document()
                doc.add_text("path", _path(page.category, page.slug))
                doc.add_text("slug", str(page.slug))
                doc.add_text("category", str(page.category))
                doc.add_text("title", str(page.title or ""))
                _add_tags(doc, page.tags)
                doc.add_text("body", str(page.body or ""))
                writer.add_document(doc)
                count += 1
            writer.commit()
        except Exception:
            writer.rollback()
            raise
        self._ix.reload()
        return count

    # ── Attach / detach ──────────────────────────────────────────
    @classmethod
    def attach(cls, wm: WikiManager, index_dir: Path | str) -> TantivyIndex:
        """Create a TantivyIndex, wire hooks, do an idempotent rebuild.

        Replaces any previously attached TantivyIndex (its hooks are
        removed first). Co-installed WhooshIndex hooks are NOT touched —
        the two backends are independent. ``wm.query_wiki()`` will route
        through this index after attach.
        """
        if cls._attached is not None:
            cls.detach()

        idx = cls(index_dir)
        try:
            idx._tenant_root = wm.root.resolve()
        except Exception:
            idx._tenant_root = wm.root
        idx.rebuild(wm.list_pages())

        wiki_manager.register_post_create(idx.add)
        wiki_manager.register_post_delete(idx.remove)
        wiki_manager.set_query_backend(idx.search)

        cls._attached = idx
        return idx

    @classmethod
    def detach(cls) -> None:
        """Unwire ONLY this index's hooks. Other plugins keep theirs."""
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
        # Bound-method equality (== not `is`) — see WhooshIndex.detach.
        if wiki_manager._query_backend == idx.search:
            wiki_manager.set_query_backend(None)
        cls._attached = None
