"""Tests for the Tantivy-backed wiki search accelerator.

Mirrors test_search_index.py so the two backends stay behavior-equivalent.
Skipped automatically if `tantivy` isn't installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

tantivy = pytest.importorskip("tantivy")

from rke.config import DEFAULTS, Config  # noqa: E402
from rke.wiki.manager import WikiManager, WikiPage, clear_hooks  # noqa: E402
from rke.wiki.tantivy_index import TantivyIndex  # noqa: E402


def _cfg(tmp_path: Path) -> Config:
    raw = {**DEFAULTS, "wiki": {"path": str(tmp_path / "wiki")}}
    return Config(raw=raw)


@pytest.fixture(autouse=True)
def _isolate_hooks():
    clear_hooks()
    yield
    TantivyIndex.detach()
    clear_hooks()


def _page(slug: str, title: str, body: str, tags: list[str] | None = None) -> WikiPage:
    return WikiPage(
        title=title, body=body, category="general",
        tags=tags or [], slug=slug,
    )


def test_index_roundtrip_title_token(tmp_path: Path):
    idx = TantivyIndex(tmp_path / "idx")
    idx.add(_page("oauth-tokens", "OAuth Tokens", "discusses refresh flows", tags=["auth"]))
    idx.add(_page("database", "Database", "postgres tuning notes", tags=["db"]))

    paths = idx.search("oauth")
    assert "general/oauth-tokens" in paths


def test_title_outranks_body_only_match(tmp_path: Path):
    idx = TantivyIndex(tmp_path / "idx")
    idx.add(_page("body-only", "Random Topic",
                  "we also mention kubernetes in passing", tags=[]))
    idx.add(_page("kube-guide", "Kubernetes Guide",
                  "nothing else special here", tags=[]))

    paths = idx.search("kubernetes", limit=5)
    assert paths, "expected at least one hit"
    assert paths[0] == "general/kube-guide"


def test_update_replaces_same_path(tmp_path: Path):
    idx = TantivyIndex(tmp_path / "idx")
    idx.add(_page("note", "Old Title", "alpha content", tags=[]))
    idx.add(_page("note", "New Title", "beta content", tags=[]))

    assert idx.search("alpha") == []
    paths = idx.search("beta")
    assert paths == ["general/note"]


def test_remove_deletes_by_slug(tmp_path: Path):
    idx = TantivyIndex(tmp_path / "idx")
    idx.add(_page("removable", "Removable", "uniquetermxyz here", tags=[]))
    assert "general/removable" in idx.search("uniquetermxyz")

    idx.remove("removable")
    assert idx.search("uniquetermxyz") == []


def test_attach_hook_populates_index_on_create(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    idx = TantivyIndex.attach(wm, tmp_path / "idx")

    wm.create_page("Kafka Primer", "streaming brokers and partitions",
                   category="entities", tags=["streaming"])

    paths = idx.search("kafka")
    assert paths == ["entities/kafka-primer"]


def test_attach_routes_query_wiki_through_backend(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    TantivyIndex.attach(wm, tmp_path / "idx")

    wm.create_page("OAuth", "discusses tokens and refresh", tags=["auth"])
    wm.create_page("Database", "postgres tuning notes", tags=["db"])

    hits = wm.query_wiki("postgres")
    assert hits and hits[0].title == "Database"

    hits2 = wm.query_wiki("refresh")
    assert hits2 and hits2[0].title == "OAuth"


def test_rebuild_wipes_and_repopulates(tmp_path: Path):
    idx = TantivyIndex(tmp_path / "idx")
    idx.add(_page("stale", "Stale Page", "outdated info", tags=[]))
    assert "general/stale" in idx.search("outdated")

    fresh = [
        _page("one", "Alpha Doc", "body about widgets", tags=["t"]),
        _page("two", "Beta Doc", "body about gadgets", tags=["t"]),
        _page("three", "Gamma Doc", "body about gizmos", tags=["t"]),
    ]
    n = idx.rebuild(fresh)
    assert n == 3

    assert idx.search("outdated") == []
    assert idx.search("widgets") == ["general/one"]
    assert idx.search("gadgets") == ["general/two"]
    assert idx.search("gizmos") == ["general/three"]


def test_remove_via_wiki_manager_hook(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    idx = TantivyIndex.attach(wm, tmp_path / "idx")

    wm.create_page("Ephemeral", "short-lived entry with flibbertigibbet", tags=[])
    assert idx.search("flibbertigibbet") == ["general/ephemeral"]

    assert wm.delete_page("Ephemeral") is True
    assert idx.search("flibbertigibbet") == []


def test_attach_twice_does_not_leak_stale_hooks(tmp_path: Path):
    from rke.wiki import manager as wm_mod

    wm1 = WikiManager(_cfg(tmp_path / "a"))
    TantivyIndex.attach(wm1, tmp_path / "idx_a")
    hooks_after_first = (
        len(wm_mod._post_create_hooks),
        len(wm_mod._post_delete_hooks),
    )

    wm2 = WikiManager(_cfg(tmp_path / "b"))
    TantivyIndex.attach(wm2, tmp_path / "idx_b")
    hooks_after_second = (
        len(wm_mod._post_create_hooks),
        len(wm_mod._post_delete_hooks),
    )
    assert hooks_after_second == hooks_after_first

    wm2.create_page("Beta", "raretokenxyz here", tags=["t"])
    hits = wm2.query_wiki("raretokenxyz")
    assert hits and hits[0].title == "Beta"


def test_same_slug_different_categories_coexist(tmp_path: Path):
    idx = TantivyIndex(tmp_path / "idx")
    a = WikiPage(title="Archive 1",
                 body="thread-A talks about flibbertigibbet things",
                 category="threads/thread-a/archive",
                 tags=["thread-archive"], slug="archive-1")
    b = WikiPage(title="Archive 1",
                 body="thread-B talks about other widgety stuff",
                 category="threads/thread-b/archive",
                 tags=["thread-archive"], slug="archive-1")
    idx.add(a)
    idx.add(b)

    only_a = idx.search("flibbertigibbet")
    assert only_a == ["threads/thread-a/archive/archive-1"]
    only_b = idx.search("widgety")
    assert only_b == ["threads/thread-b/archive/archive-1"]

    idx.remove(a)
    after_a = idx.search("flibbertigibbet")
    assert after_a == []
    after_b = idx.search("widgety")
    assert after_b == ["threads/thread-b/archive/archive-1"]


def test_detach_clears_query_backend(tmp_path: Path):
    """Bound-method == comparison must clear the backend cleanly."""
    from rke.wiki import manager as wm_mod

    wm = WikiManager(_cfg(tmp_path))
    TantivyIndex.attach(wm, tmp_path / "idx")
    assert wm_mod._query_backend is not None
    TantivyIndex.detach()
    assert wm_mod._query_backend is None
