"""Tests for rke.wiki.lifecycle (TTL / LRU / access tracking)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from rke.config import DEFAULTS, Config
from rke.wiki.lifecycle import (
    AccessTracker,
    evict_expired,
    evict_lru,
    expired_pages,
    is_expired,
    lru_candidates,
    set_expiry,
    touch,
)
from rke.wiki.manager import WikiManager


def _cfg(tmp_path: Path) -> Config:
    raw = {**DEFAULTS, "wiki": {"path": str(tmp_path / "wiki")}}
    return Config(raw=raw)


def _wm(tmp_path: Path) -> WikiManager:
    return WikiManager(_cfg(tmp_path))


def test_set_expiry_days_sets_future_timestamp(tmp_path: Path):
    wm = _wm(tmp_path)
    wm.create_page("Alpha", "body", tags=["t"])
    before = datetime.now(timezone.utc)
    page = set_expiry(wm, "alpha", days=7)
    after = datetime.now(timezone.utc)
    assert page is not None
    assert page.expires_at
    parsed = datetime.fromisoformat(page.expires_at)
    # Should be roughly 7 days out.
    assert before + timedelta(days=7) - timedelta(seconds=5) <= parsed
    assert parsed <= after + timedelta(days=7) + timedelta(seconds=5)
    # Body / tags survive round-trip.
    reread = wm.get_page("alpha")
    assert reread is not None
    assert reread.expires_at == page.expires_at
    assert reread.tags == ["t"]
    assert "body" in reread.body


def test_set_expiry_unknown_slug_returns_none(tmp_path: Path):
    wm = _wm(tmp_path)
    assert set_expiry(wm, "does-not-exist", days=1) is None


def test_is_expired_false_when_unset(tmp_path: Path):
    wm = _wm(tmp_path)
    wm.create_page("Beta", "b", tags=["x"])
    page = wm.get_page("beta")
    assert page is not None
    assert page.expires_at == ""
    assert is_expired(page) is False


def test_is_expired_true_when_past(tmp_path: Path):
    wm = _wm(tmp_path)
    wm.create_page("Gamma", "g", tags=["x"])
    past = datetime.now(timezone.utc) - timedelta(days=1)
    page = set_expiry(wm, "gamma", at=past)
    assert page is not None
    assert is_expired(page) is True
    # Unparseable value -> treated as missing, NOT expired.
    page.expires_at = "not-a-date"
    assert is_expired(page) is False


def test_touch_updates_last_accessed_at(tmp_path: Path):
    wm = _wm(tmp_path)
    wm.create_page("Delta", "d", tags=["x"])
    before = wm.get_page("delta")
    assert before is not None and before.last_accessed_at == ""

    updated = touch(wm, "delta")
    assert updated is not None
    assert updated.last_accessed_at
    parsed = datetime.fromisoformat(updated.last_accessed_at)
    # Within the last few seconds.
    now = datetime.now(timezone.utc)
    assert now - parsed < timedelta(seconds=5)

    # Unknown slug -> None.
    assert touch(wm, "nope") is None


def test_expired_pages_collects_only_past_due(tmp_path: Path):
    wm = _wm(tmp_path)
    wm.create_page("Fresh", "f", tags=["x"])
    wm.create_page("Stale", "s", tags=["x"])
    wm.create_page("NoExpiry", "n", tags=["x"])

    future = datetime.now(timezone.utc) + timedelta(days=30)
    past = datetime.now(timezone.utc) - timedelta(days=1)
    set_expiry(wm, "fresh", at=future)
    set_expiry(wm, "stale", at=past)

    slugs = {p.slug for p in expired_pages(wm)}
    assert slugs == {"stale"}


def test_evict_expired_deletes_files(tmp_path: Path):
    wm = _wm(tmp_path)
    wm.create_page("Keeper", "k", tags=["x"])
    wm.create_page("Doomed", "d", tags=["x"])
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    set_expiry(wm, "doomed", at=past)

    doomed_path = wm.get_page("doomed").path  # type: ignore[union-attr]
    assert doomed_path is not None and doomed_path.exists()

    evicted = evict_expired(wm)
    assert evicted == ["doomed"]
    assert not doomed_path.exists()
    assert wm.get_page("doomed") is None
    assert wm.get_page("keeper") is not None


def test_lru_candidates_pages_without_access_sort_first(tmp_path: Path):
    wm = _wm(tmp_path)
    wm.create_page("Old", "o", tags=["x"])
    wm.create_page("Newer", "n", tags=["x"])
    wm.create_page("Never", "v", tags=["x"])

    # Manually set last_accessed_at on two of them.
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    new_ts = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()

    old_page = wm.get_page("old")
    assert old_page is not None
    old_page.last_accessed_at = old_ts
    assert old_page.path is not None
    old_page.path.write_text(old_page.to_markdown(), encoding="utf-8")

    new_page = wm.get_page("newer")
    assert new_page is not None
    new_page.last_accessed_at = new_ts
    assert new_page.path is not None
    new_page.path.write_text(new_page.to_markdown(), encoding="utf-8")

    ordered = lru_candidates(wm, keep_last_n=0)
    slugs = [p.slug for p in ordered]
    assert slugs[0] == "never"          # no last_accessed_at → oldest
    assert slugs.index("old") < slugs.index("newer")


def test_evict_lru_keeps_exactly_keep_last_n(tmp_path: Path):
    wm = _wm(tmp_path)
    base = datetime.now(timezone.utc) - timedelta(days=5)
    for i in range(5):
        wm.create_page(f"Page {i}", f"body{i}", tags=["x"])
        page = wm.get_page(f"page-{i}")
        assert page is not None and page.path is not None
        page.last_accessed_at = (base + timedelta(hours=i)).isoformat()
        page.path.write_text(page.to_markdown(), encoding="utf-8")

    evicted = evict_lru(wm, keep_last_n=2)
    assert len(evicted) == 3
    remaining = wm.list_pages()
    assert len(remaining) == 2
    remaining_slugs = {p.slug for p in remaining}
    # The two most-recently accessed (indexes 3 and 4) survive.
    assert remaining_slugs == {"page-3", "page-4"}


def test_access_tracker_install_uninstall(tmp_path: Path):
    wm = _wm(tmp_path)
    wm.create_page("Tracked", "t", tags=["x"])

    # Initial state: no last_accessed_at.
    raw = wm.get_page("tracked")
    assert raw is not None and raw.last_accessed_at == ""

    # Snapshot the underlying bound method before patching.
    original_bound = wm.get_page
    tracker = AccessTracker(wm)
    tracker.install()
    # Idempotent second install.
    tracker.install()
    # Now get_page is an instance attribute (the wrapper), not the class
    # method.  Verify it is not the same object as the original bound method.
    assert wm.get_page is not original_bound

    fetched = wm.get_page("tracked")
    assert fetched is not None
    # Reload via the saved original to see the updated timestamp the
    # tracker just wrote to disk.
    reread = original_bound("tracked")
    assert reread is not None
    assert reread.last_accessed_at != ""

    tracker.uninstall()
    # After uninstall the instance attribute is gone — get_page resolves
    # back through the class.
    assert "get_page" not in wm.__dict__

    # After uninstall, touching should not happen automatically.  Clear
    # the timestamp and confirm a read leaves it clear.
    cleared = wm.get_page("tracked")
    assert cleared is not None and cleared.path is not None
    cleared.last_accessed_at = ""
    cleared.path.write_text(cleared.to_markdown(), encoding="utf-8")

    after = wm.get_page("tracked")
    assert after is not None
    assert after.last_accessed_at == ""
