"""LLM-Wiki manager — git-tracked markdown knowledge base.

Pages live under `wiki.path/<category>/<slug>.md`. Each page has a YAML
frontmatter block followed by the markdown body.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..config import Config, load_config

log = logging.getLogger(__name__)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class WikiPage:
    title: str
    body: str
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    slug: str = ""
    path: Path | None = None
    created_at: str = ""
    updated_at: str = ""
    # Optional lifecycle fields (used by rke.wiki.lifecycle). Empty string = unset.
    expires_at: str = ""
    last_accessed_at: str = ""

    def to_markdown(self) -> str:
        meta: dict[str, Any] = {
            "title": self.title,
            "category": self.category,
            "tags": list(self.tags),
            "slug": self.slug,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.expires_at:
            meta["expires_at"] = self.expires_at
        if self.last_accessed_at:
            meta["last_accessed_at"] = self.last_accessed_at
        return f"---\n{yaml.safe_dump(meta, sort_keys=False).strip()}\n---\n\n{self.body.lstrip()}\n"


# ── Extension hooks ────────────────────────────────────────────
# Modules can register callbacks that fire after create/delete. Used by:
#   rke.wiki.search_index  — keep an inverted index in sync
#   rke.wiki.lifecycle     — touch last_accessed_at, evict on TTL
#   rke.knowledge.extractor — pull entities/relations from new pages
_post_create_hooks: list = []
_post_delete_hooks: list = []


def register_post_create(fn) -> None:
    """Register a callable invoked as fn(page: WikiPage) after each create_page."""
    if fn not in _post_create_hooks:
        _post_create_hooks.append(fn)


def register_post_delete(fn) -> None:
    """Register a callable invoked as fn(slug: str) after each delete_page."""
    if fn not in _post_delete_hooks:
        _post_delete_hooks.append(fn)


def clear_hooks() -> None:
    """Test helper: drop all registered hooks."""
    _post_create_hooks.clear()
    _post_delete_hooks.clear()
    global _query_backend
    _query_backend = None


# Optional pluggable accelerator for query_wiki(). A backend is a callable
# (query: str, limit: int) -> list[str] returning matching slugs, ranked.
_query_backend = None


def set_query_backend(fn) -> None:
    """Install a query accelerator (e.g. a Whoosh index)."""
    global _query_backend
    _query_backend = fn


@dataclass
class LintFinding:
    kind: str  # "orphan", "broken_link", "duplicate_title", "empty"
    page: str
    detail: str


def slugify(value: str) -> str:
    """ASCII slug — ' My Topic! ' → 'my-topic'."""
    s = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return s or "untitled"


class WikiManager:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.root = self.config.wiki_path
        self.root.mkdir(parents=True, exist_ok=True)

    # ── CRUD ─────────────────────────────────────────────────────
    def create_page(
        self,
        title: str,
        body: str,
        category: str = "general",
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> WikiPage:
        slug = slugify(title)
        cat_dir = self.root / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        path = cat_dir / f"{slug}.md"
        now = datetime.now(timezone.utc).isoformat()

        if path.exists() and not overwrite:
            existing = self._read_page(path)
            page = WikiPage(
                title=title,
                body=body,
                category=category,
                tags=tags or existing.tags,
                slug=slug,
                path=path,
                created_at=existing.created_at or now,
                updated_at=now,
                # Preserve lifecycle frontmatter across updates so a normal
                # body edit does not silently disable TTL/LRU bookkeeping.
                expires_at=existing.expires_at,
                last_accessed_at=existing.last_accessed_at,
            )
        else:
            page = WikiPage(
                title=title,
                body=body,
                category=category,
                tags=tags or [],
                slug=slug,
                path=path,
                created_at=now,
                updated_at=now,
            )
        path.write_text(page.to_markdown(), encoding="utf-8")
        log.info("wiki: wrote %s", path)
        for hook in list(_post_create_hooks):
            try:
                hook(page)
            except Exception as exc:
                log.warning("post_create hook %r failed: %s", hook, exc)
        return page

    def get_page(self, slug_or_title: str, category: str | None = None) -> WikiPage | None:
        slug = slugify(slug_or_title)
        if category:
            candidate = self.root / category / f"{slug}.md"
            if candidate.exists():
                return self._read_page(candidate)
        for path in self.root.rglob(f"{slug}.md"):
            return self._read_page(path)
        return None

    def list_pages(self, category: str | None = None) -> list[WikiPage]:
        base = self.root / category if category else self.root
        if not base.exists():
            return []
        return [self._read_page(p) for p in sorted(base.rglob("*.md"))]

    def delete_page(self, slug_or_title: str, category: str | None = None) -> bool:
        page = self.get_page(slug_or_title, category)
        if page and page.path and page.path.exists():
            page.path.unlink()
            for hook in list(_post_delete_hooks):
                try:
                    # Pass the full page so consumers (e.g. search_index) can
                    # disambiguate same-slug-different-category entries.
                    hook(page)
                except Exception as exc:
                    log.warning("post_delete hook %r failed: %s", hook, exc)
            return True
        return False

    # ── Search / query ───────────────────────────────────────────
    def query_wiki(self, query: str, limit: int = 5) -> list[WikiPage]:
        """Keyword search.

        If `rke.wiki.search_index` has been imported and registered an
        accelerator via `set_query_backend`, use it; otherwise fall back to
        the deterministic linear substring scan.
        """
        if _query_backend is not None:
            try:
                ids = _query_backend(query, limit)
                pages: list[WikiPage] = []
                for ident in ids:
                    # Backends may return either "slug" or "category/slug"
                    # composite identifiers. Composite form lets us look up
                    # the exact page when the same slug exists in multiple
                    # categories (e.g. chat-thread archive pages).
                    if "/" in ident:
                        category, slug = ident.rsplit("/", 1)
                    else:
                        category, slug = None, ident
                    p = self.get_page(slug, category=category)
                    if p:
                        pages.append(p)
                if pages:
                    return pages
            except Exception as exc:
                log.warning("query_backend failed (%s); using linear scan", exc)

        q = query.lower()
        scored: list[tuple[int, WikiPage]] = []
        for page in self.list_pages():
            score = 0
            if q in page.title.lower():
                score += 5
            score += sum(2 for t in page.tags if q in t.lower())
            score += page.body.lower().count(q)
            if score:
                scored.append((score, page))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]

    def index(self) -> dict[str, dict[str, Any]]:
        """Return a {slug: {title, category, tags}} index — cheap to compute."""
        out: dict[str, dict[str, Any]] = {}
        for page in self.list_pages():
            out[page.slug] = {
                "title": page.title,
                "category": page.category,
                "tags": page.tags,
                "path": str(page.path),
            }
        return out

    # ── Health / lint ────────────────────────────────────────────
    def lint(self) -> list[LintFinding]:
        findings: list[LintFinding] = []
        pages = self.list_pages()
        seen_titles: dict[str, str] = {}
        slug_set = {p.slug for p in pages}
        for p in pages:
            if not p.body.strip():
                findings.append(LintFinding("empty", p.slug, "page body is empty"))
            if p.title in seen_titles and seen_titles[p.title] != p.slug:
                findings.append(LintFinding(
                    "duplicate_title", p.slug,
                    f"title '{p.title}' also used by {seen_titles[p.title]}",
                ))
            seen_titles.setdefault(p.title, p.slug)
            # broken internal links of the form [text](some-slug)
            for match in re.finditer(r"\]\(([a-z0-9\-]+)\)", p.body):
                target = match.group(1)
                if target not in slug_set and not target.startswith(("http", "/")):
                    findings.append(LintFinding(
                        "broken_link", p.slug,
                        f"link to unknown slug '{target}'",
                    ))
        # orphans: no incoming links and no tags (rough heuristic)
        link_targets: set[str] = set()
        for p in pages:
            link_targets.update(re.findall(r"\]\(([a-z0-9\-]+)\)", p.body))
        for p in pages:
            if not p.tags and p.slug not in link_targets:
                findings.append(LintFinding("orphan", p.slug, "no tags and no inbound links"))
        return findings

    # ── Internal ─────────────────────────────────────────────────
    def _read_page(self, path: Path) -> WikiPage:
        text = path.read_text(encoding="utf-8")
        meta: dict[str, Any] = {}
        body = text
        m = _FRONTMATTER_RE.match(text)
        if m:
            try:
                meta = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError:
                meta = {}
            body = text[m.end():]
        category = path.parent.relative_to(self.root).as_posix() or "general"
        return WikiPage(
            title=meta.get("title", path.stem),
            body=body,
            category=meta.get("category", category),
            tags=list(meta.get("tags", [])),
            slug=meta.get("slug", path.stem),
            path=path,
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
            expires_at=meta.get("expires_at", ""),
            last_accessed_at=meta.get("last_accessed_at", ""),
        )
