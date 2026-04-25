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

    slugs = idx.search("oauth")
    assert "oauth-tokens" in slugs


def test_title_outranks_body_only_match(tmp_path: Path):
    idx = WhooshIndex(tmp_path / "idx")
    # "kubernetes" only in body
    idx.add(_page("body-only", "Random Topic",
                  "we also mention kubernetes in passing", tags=[]))
    # "kubernetes" in title — should win via 5x title boost
    idx.add(_page("kube-guide", "Kubernetes Guide",
                  "nothing else special here", tags=[]))

    slugs = idx.search("kubernetes", limit=5)
    assert slugs, "expected at least one hit"
    assert slugs[0] == "kube-guide"


def test_update_replaces_same_slug(tmp_path: Path):
    idx = WhooshIndex(tmp_path / "idx")
    idx.add(_page("note", "Old Title", "alpha content", tags=[]))
    idx.add(_page("note", "New Title", "beta content", tags=[]))

    # Old term no longer present
    assert idx.search("alpha") == []
    # New term finds the single (updated) doc
    slugs = idx.search("beta")
    assert slugs == ["note"]


def test_remove_deletes_slug(tmp_path: Path):
    idx = WhooshIndex(tmp_path / "idx")
    idx.add(_page("removable", "Removable", "unique-term-xyz here", tags=[]))
    assert "removable" in idx.search("unique-term-xyz")

    idx.remove("removable")
    assert idx.search("unique-term-xyz") == []


def test_attach_hook_populates_index_on_create(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    idx = WhooshIndex.attach(wm, tmp_path / "idx")

    wm.create_page("Kafka Primer", "streaming brokers and partitions",
                   category="entities", tags=["streaming"])

    slugs = idx.search("kafka")
    assert slugs == ["kafka-primer"]


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
    assert "stale" in idx.search("outdated")

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
    assert idx.search("widgets") == ["one"]
    assert idx.search("gadgets") == ["two"]
    assert idx.search("gizmos") == ["three"]


def test_remove_via_wiki_manager_hook(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    idx = WhooshIndex.attach(wm, tmp_path / "idx")

    wm.create_page("Ephemeral", "short-lived entry with flibbertigibbet", tags=[])
    assert idx.search("flibbertigibbet") == ["ephemeral"]

    assert wm.delete_page("Ephemeral") is True
    assert idx.search("flibbertigibbet") == []
