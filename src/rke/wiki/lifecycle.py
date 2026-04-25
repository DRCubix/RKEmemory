"""Letta-style memory pressure / TTL layer for the wiki.

Pages carry two optional frontmatter fields (already on WikiPage):

* ``expires_at``         — ISO-8601 UTC timestamp. Past-due pages are
                           candidates for eviction.
* ``last_accessed_at``   — ISO-8601 UTC timestamp updated on read.  Used
                           by LRU-style policies.

Everything here is additive: pages without lifecycle metadata are
immortal, and the module never mutates existing behaviour unless an
AccessTracker has been explicitly installed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .manager import WikiManager, WikiPage

# ── helpers ─────────────────────────────────────────────────────────────


def now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _utcnow(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _parse(ts: str) -> datetime | None:
    """Parse an ISO-8601 string, returning None on failure or empty input."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _rewrite(wm: WikiManager, page: WikiPage) -> WikiPage:
    """Re-save ``page`` preserving ALL metadata (including created_at and
    lifecycle fields). Writes directly via ``WikiPage.to_markdown()`` rather
    than routing through ``wm.create_page(overwrite=True)`` because the
    latter resets ``created_at`` and re-fires post_create hooks on every
    lifecycle change."""
    if page.path is None:
        # Page was never persisted — fall back to create_page to give it a path.
        return wm.create_page(
            title=page.title,
            body=page.body,
            category=page.category,
            tags=list(page.tags),
        )
    page.updated_at = now_iso()
    page.path.write_text(page.to_markdown(), encoding="utf-8")
    return page


# ── public API ──────────────────────────────────────────────────────────


def set_expiry(
    wm: WikiManager,
    slug: str,
    *,
    days: int | None = None,
    at: datetime | None = None,
) -> WikiPage | None:
    """Set (or clear) the ``expires_at`` frontmatter on an existing page.

    Pass exactly one of ``days`` or ``at``.  Returns the re-saved page, or
    ``None`` if the slug is unknown.
    """
    if days is not None and at is not None:
        raise ValueError("pass days OR at, not both")
    if days is None and at is None:
        raise ValueError("must pass days or at")

    page = wm.get_page(slug)
    if page is None:
        return None

    if at is not None:
        expiry = _utcnow(at)
    else:
        expiry = datetime.now(timezone.utc) + timedelta(days=days or 0)

    page.expires_at = expiry.isoformat()
    return _rewrite(wm, page)


def touch(wm: WikiManager, slug: str) -> WikiPage | None:
    """Update ``last_accessed_at`` to now and re-save.  Returns the page."""
    page = wm.get_page(slug)
    if page is None:
        return None
    page.last_accessed_at = now_iso()
    return _rewrite(wm, page)


def is_expired(page: WikiPage, now: datetime | None = None) -> bool:
    """True iff ``page.expires_at`` is parseable and <= now."""
    expiry = _parse(page.expires_at)
    if expiry is None:
        return False
    return expiry <= _utcnow(now)


def expired_pages(
    wm: WikiManager, now: datetime | None = None
) -> list[WikiPage]:
    """Return all pages whose ``expires_at`` is in the past."""
    cutoff = _utcnow(now)
    return [p for p in wm.list_pages() if is_expired(p, cutoff)]


def evict_expired(
    wm: WikiManager, now: datetime | None = None
) -> list[str]:
    """Delete every expired page via ``wm.delete_page``.  Returns slugs."""
    evicted: list[str] = []
    for page in expired_pages(wm, now):
        if wm.delete_page(page.slug, category=page.category):
            evicted.append(page.slug)
    return evicted


def lru_candidates(
    wm: WikiManager, *, keep_last_n: int = 100
) -> list[WikiPage]:
    """Pages sorted oldest-access-first, excluding the ``keep_last_n`` freshest.

    Pages with no ``last_accessed_at`` sort BEFORE pages that have one —
    they are considered the oldest (never read).
    """
    pages = wm.list_pages()

    def key(p: WikiPage) -> tuple[int, datetime]:
        dt = _parse(p.last_accessed_at)
        if dt is None:
            # Sort before any real datetime.
            return (0, datetime.min.replace(tzinfo=timezone.utc))
        return (1, dt)

    pages.sort(key=key)
    if keep_last_n <= 0:
        return pages
    if keep_last_n >= len(pages):
        return []
    return pages[: len(pages) - keep_last_n]


def evict_lru(wm: WikiManager, *, keep_last_n: int) -> list[str]:
    """Evict pages until at most ``keep_last_n`` remain.  Returns slugs."""
    evicted: list[str] = []
    for page in lru_candidates(wm, keep_last_n=keep_last_n):
        if wm.delete_page(page.slug, category=page.category):
            evicted.append(page.slug)
    return evicted


# ── access tracker ──────────────────────────────────────────────────────


class AccessTracker:
    """Monkey-patch ``wm.get_page`` so every successful read touches the page.

    Usage::

        tracker = AccessTracker(wm)
        tracker.install()
        ...
        tracker.uninstall()

    Installing twice is a no-op.  Uninstall restores the original method.
    """

    _ATTR = "_lifecycle_original_get_page"

    def __init__(self, wm: WikiManager) -> None:
        self.wm = wm

    def install(self) -> None:
        if getattr(self.wm, self._ATTR, None) is not None:
            return  # idempotent
        original = self.wm.get_page
        setattr(self.wm, self._ATTR, original)

        def wrapper(slug_or_title: str, category: str | None = None):
            page = original(slug_or_title, category)
            if page is not None and page.path is not None:
                try:
                    # Inlined touch — must NOT re-enter wm.get_page, otherwise
                    # the wrapper would recurse indefinitely (each access would
                    # trigger another access). Update in-place and write the
                    # file directly.
                    page.last_accessed_at = now_iso()
                    page.updated_at = now_iso()
                    page.path.write_text(page.to_markdown(), encoding="utf-8")
                except Exception:
                    # Never let lifecycle bookkeeping break reads.
                    pass
            return page

        self.wm.get_page = wrapper  # type: ignore[method-assign]

    def uninstall(self) -> None:
        original = getattr(self.wm, self._ATTR, None)
        if original is None:
            return
        # Remove the instance-level wrapper so that attribute lookup falls
        # back to the class-bound get_page again.
        if "get_page" in self.wm.__dict__:
            del self.wm.__dict__["get_page"]
        delattr(self.wm, self._ATTR)

    def __enter__(self) -> AccessTracker:
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.uninstall()
