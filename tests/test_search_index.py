"""Tests for the Whoosh-backed wiki search accelerator."""

from __future__ import annotations

from pathlib import Path

import pytest

from rke.config import DEFAULTS, Config
from rke.wiki.manager import WikiManager, WikiPage, clear_hooks
from rke.wiki.search_index import WhooshIndex


def _cfg(tmp_path: Path) -> Config:
    raw = {**DEFAULTS, "wiki": {"path": str(tmp_path / "wiki")}}
    return Config(raw=raw)


@pytest.fixture(autouse=True)
def _isolate_hooks():
    """Every test starts with a clean hook registry / query backend."""
    clear_hooks()
    yield
    WhooshIndex.detach()
    clear_hooks()


def _page(slug: str, title: str, body: str, tags: list[str] | None = None) -> WikiPage:
    return WikiPage(
        title=title,
        body=body,
        category="general",
        tags=tags or [],
        slug=slug,
    )


def test_index_roundtrip_title_token(tmp_path: Path):
    idx = WhooshIndex(tmp_path / "idx")
    idx.add(_page("oauth-tokens", "OAuth Tokens", "discusses refresh flows", tags=["auth"]))
    idx.add(_page("database", "Database", "postgres tuning notes", tags=["db"]))

    paths = idx.search("oauth")
    assert "general/oauth-tokens" in paths


def test_title_outranks_body_only_match(tmp_path: Path):
    idx = WhooshIndex(tmp_path / "idx")
    # "kubernetes" only in body
    idx.add(_page("body-only", "Random Topic",
                  "we also mention kubernetes in passing", tags=[]))
    # "kubernetes" in title — should win via 5x title boost
    idx.add(_page("kube-guide", "Kubernetes Guide",
                  "nothing else special here", tags=[]))

    paths = idx.search("kubernetes", limit=5)
    assert paths, "expected at least one hit"
    assert paths[0] == "general/kube-guide"


def test_update_replaces_same_slug(tmp_path: Path):
    idx = WhooshIndex(tmp_path / "idx")
    idx.add(_page("note", "Old Title", "alpha content", tags=[]))
    idx.add(_page("note", "New Title", "beta content", tags=[]))

    # Old term no longer present
    assert idx.search("alpha") == []
    # New term finds the single (updated) doc
    paths = idx.search("beta")
    assert paths == ["general/note"]


def test_remove_deletes_slug(tmp_path: Path):
    idx = WhooshIndex(tmp_path / "idx")
    idx.add(_page("removable", "Removable", "unique-term-xyz here", tags=[]))
    assert "general/removable" in idx.search("unique-term-xyz")

    idx.remove("removable")
    assert idx.search("unique-term-xyz") == []


def test_attach_hook_populates_index_on_create(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    idx = WhooshIndex.attach(wm, tmp_path / "idx")

    wm.create_page("Kafka Primer", "streaming brokers and partitions",
                   category="entities", tags=["streaming"])

    paths = idx.search("kafka")
    assert paths == ["entities/kafka-primer"]


def test_attach_routes_query_wiki_through_backend(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    WhooshIndex.attach(wm, tmp_path / "idx")

    wm.create_page("OAuth", "discusses tokens and refresh", tags=["auth"])
    wm.create_page("Database", "postgres tuning notes", tags=["db"])

    hits = wm.query_wiki("postgres")
    assert hits and hits[0].title == "Database"

    hits2 = wm.query_wiki("refresh")
    assert hits2 and hits2[0].title == "OAuth"


def test_rebuild_wipes_and_repopulates(tmp_path: Path):
    idx = WhooshIndex(tmp_path / "idx")
    idx.add(_page("stale", "Stale Page", "outdated info", tags=[]))
    assert "general/stale" in idx.search("outdated")

    fresh = [
        _page("one", "Alpha Doc", "body about widgets", tags=["t"]),
        _page("two", "Beta Doc", "body about gadgets", tags=["t"]),
        _page("three", "Gamma Doc", "body about gizmos", tags=["t"]),
    ]
    n = idx.rebuild(fresh)
    assert n == 3

    # Old doc is gone
    assert idx.search("outdated") == []
    # New docs searchable
    assert idx.search("widgets") == ["general/one"]
    assert idx.search("gadgets") == ["general/two"]
    assert idx.search("gizmos") == ["general/three"]


def test_remove_via_wiki_manager_hook(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    idx = WhooshIndex.attach(wm, tmp_path / "idx")

    wm.create_page("Ephemeral", "short-lived entry with flibbertigibbet", tags=[])
    assert idx.search("flibbertigibbet") == ["general/ephemeral"]

    assert wm.delete_page("Ephemeral") is True
    assert idx.search("flibbertigibbet") == []


def test_attach_twice_does_not_leak_stale_hooks(tmp_path: Path):
    """Regression: Codex caught that attach() over an already-attached
    instance left the previous index's hooks registered."""
    from rke.wiki import manager as wm_mod

    wm1 = WikiManager(_cfg(tmp_path / "a"))
    WhooshIndex.attach(wm1, tmp_path / "idx_a")
    hooks_after_first = (
        len(wm_mod._post_create_hooks),
        len(wm_mod._post_delete_hooks),
    )

    wm2 = WikiManager(_cfg(tmp_path / "b"))
    WhooshIndex.attach(wm2, tmp_path / "idx_b")
    hooks_after_second = (
        len(wm_mod._post_create_hooks),
        len(wm_mod._post_delete_hooks),
    )

    # The second attach should NOT add a new pair on top of the first —
    # it should replace the previous index.
    assert hooks_after_second == hooks_after_first

    # And the surviving query backend should serve the second WikiManager.
    wm2.create_page("Beta", "rare-token-xyz here", tags=["t"])
    hits = wm2.query_wiki("rare-token-xyz")
    assert hits and hits[0].title == "Beta"


def test_same_slug_different_categories_coexist(tmp_path: Path):
    """Regression: Codex caught that two pages with identical slugs in
    different categories (e.g. chat-thread archive pages all named
    'archive-1') were overwriting each other in the index because the
    Whoosh schema's unique key was just `slug`."""
    idx = WhooshIndex(tmp_path / "idx")

    a = WikiPage(title="Archive 1", body="thread-A talks about flibbertigibbet things",
                 category="threads/thread-a/archive", tags=["thread-archive"], slug="archive-1")
    b = WikiPage(title="Archive 1", body="thread-B talks about other widgety stuff",
                 category="threads/thread-b/archive", tags=["thread-archive"], slug="archive-1")
    idx.add(a)
    idx.add(b)

    paths = idx.search("archive 1", limit=5)
    # BOTH pages must survive in the index — neither overwrote the other.
    assert "threads/thread-a/archive/archive-1" in paths
    assert "threads/thread-b/archive/archive-1" in paths

    # Searching by a body-unique token resolves to exactly the right page.
    only_a = idx.search("flibbertigibbet")
    assert only_a == ["threads/thread-a/archive/archive-1"]
    only_b = idx.search("widgety")
    assert only_b == ["threads/thread-b/archive/archive-1"]

    # Removing one (by passing the page) does not affect the other.
    idx.remove(a)
    after = idx.search("archive 1", limit=5)
    assert "threads/thread-a/archive/archive-1" not in after
    assert "threads/thread-b/archive/archive-1" in after


def test_query_wiki_resolves_composite_paths(tmp_path: Path):
    """Verify WikiManager.query_wiki splits 'category/slug' from the backend
    and looks up the right page even when slugs collide across categories."""
    wm = WikiManager(_cfg(tmp_path))
    WhooshIndex.attach(wm, tmp_path / "idx")

    # Use distinctive tokens that won't get split by Whoosh's tokenizer.
    wm.create_page("Note", "alphaonlymarkerxyz here", category="cat-alpha")
    wm.create_page("Note", "betaonlymarkerwxv here", category="cat-beta")

    pages = wm.query_wiki("alphaonlymarkerxyz")
    assert pages and pages[0].category == "cat-alpha"
    pages2 = wm.query_wiki("betaonlymarkerwxv")
    assert pages2 and pages2[0].category == "cat-beta"
