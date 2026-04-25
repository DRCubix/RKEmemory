"""Tests for WikiManager (no external services required)."""

from __future__ import annotations

from pathlib import Path

from rke.config import DEFAULTS, Config
from rke.wiki.manager import WikiManager, slugify


def _cfg(tmp_path: Path) -> Config:
    raw = {**DEFAULTS, "wiki": {"path": str(tmp_path / "wiki")}}
    return Config(raw=raw)


def test_slugify():
    assert slugify("My Topic!") == "my-topic"
    assert slugify("  Multi  Space  ") == "multi-space"
    assert slugify("Already-Slug") == "already-slug"
    assert slugify("") == "untitled"


def test_create_get_list(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    page = wm.create_page("Auth Patterns", "# Auth\n\nbody", category="entities", tags=["auth"])
    assert page.slug == "auth-patterns"
    assert page.path is not None and page.path.exists()

    fetched = wm.get_page("Auth Patterns")
    assert fetched is not None
    assert fetched.title == "Auth Patterns"
    assert "auth" in fetched.tags

    pages = wm.list_pages()
    assert len(pages) == 1


def test_update_overwrites_body(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    wm.create_page("Topic", "# v1\noriginal", category="general", tags=["t"])
    wm.create_page("Topic", "# v2\nupdated", category="general")
    page = wm.get_page("Topic")
    assert page is not None
    assert "updated" in page.body
    assert "original" not in page.body
    # tags preserved when not passed on update
    assert page.tags == ["t"]


def test_query_wiki_keyword(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    wm.create_page("OAuth", "discusses tokens and refresh", tags=["auth"])
    wm.create_page("Database", "postgres tuning notes", tags=["db"])
    hits = wm.query_wiki("token")
    assert hits and hits[0].title == "OAuth"
    hits2 = wm.query_wiki("postgres")
    assert hits2 and hits2[0].title == "Database"


def test_lint_finds_empty_and_orphan(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    wm.create_page("Empty", "", tags=[])
    wm.create_page("Orphan", "body", tags=[])
    findings = wm.lint()
    kinds = {f.kind for f in findings}
    assert "empty" in kinds
    assert "orphan" in kinds


def test_lint_finds_broken_link(tmp_path: Path):
    wm = WikiManager(_cfg(tmp_path))
    wm.create_page("Linker", "see [other](nonexistent-slug) please", tags=["x"])
    findings = wm.lint()
    broken = [f for f in findings if f.kind == "broken_link"]
    assert broken
    assert "nonexistent-slug" in broken[0].detail
