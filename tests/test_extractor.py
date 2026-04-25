"""Tests for the knowledge extractor (offline, no LLM, no docker)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rke.config import DEFAULTS, Config
from rke.knowledge.extractor import (
    EntityExtractor,
    ExtractedEntity,
    attach_to_wiki,
)
from rke.wiki.manager import WikiManager, clear_hooks


def _cfg(tmp_path: Path) -> Config:
    raw = {**DEFAULTS, "wiki": {"path": str(tmp_path / "wiki")}}
    return Config(raw=raw)


@pytest.fixture(autouse=True)
def _wipe_hooks():
    """Each test gets a clean slate of wiki hooks."""
    clear_hooks()
    yield
    clear_hooks()


# ── Regex backend ────────────────────────────────────────────────

def test_regex_extracts_proper_noun_entity():
    ex = EntityExtractor()
    entities, _ = ex.extract("Project Alpha is our main initiative.")
    names = {e.name for e in entities}
    assert "Project Alpha" in names


def test_regex_picks_up_uses_relation():
    ex = EntityExtractor()
    entities, relations = ex.extract(
        "Project Alpha uses Qdrant. Qdrant depends on Rocksdb."
    )
    rel_triples = {(r.src_name, r.rel_type, r.dst_name) for r in relations}
    assert ("Project Alpha", "USES", "Qdrant") in rel_triples
    assert ("Qdrant", "DEPENDS_ON", "Rocksdb") in rel_triples

    names = {e.name for e in entities}
    assert "Project Alpha" in names
    assert "Qdrant" in names
    assert "Rocksdb" in names


def test_regex_filters_stopwords_and_lowercase_starts():
    ex = EntityExtractor()
    entities, _ = ex.extract(
        "The quick brown fox jumps. When a project ships, it works. "
        "This is important. Acme Corp also ships."
    )
    names = {e.name for e in entities}
    # Stopwords / sentence starts should be excluded.
    assert "The" not in names
    assert "When" not in names
    assert "This" not in names
    assert "A" not in names
    # Lower-case words are never matched by the noun regex.
    assert "quick" not in names
    assert "fox" not in names
    # A real proper noun still makes it through.
    assert "Acme Corp" in names


def test_regex_caps_entities_at_30():
    # Generate 50 distinct capitalised names; extractor must cap at 30.
    # Separate with periods so the proper-noun regex treats each as its
    # own candidate instead of greedily merging into 3-word runs.
    text = ". ".join(f"Entity{i}" for i in range(50)) + "."
    ex = EntityExtractor()
    entities, _ = ex.extract(text)
    assert len(entities) == 30


# ── attach_to_wiki ───────────────────────────────────────────────

def test_attach_to_wiki_triggers_add_entity(tmp_path: Path):
    graph = MagicMock()
    attach_to_wiki(graph)

    wm = WikiManager(_cfg(tmp_path))
    wm.create_page(
        "Alpha Notes",
        "Project Alpha uses Qdrant. Qdrant depends on Rocksdb.",
        category="notes",
    )

    assert graph.add_entity.called, "graph.add_entity should be invoked"
    added_names = {
        call.args[0].name for call in graph.add_entity.call_args_list
    }
    assert "Project Alpha" in added_names
    assert "Qdrant" in added_names

    assert graph.add_relation.called, "graph.add_relation should be invoked"
    rel_triples = {
        (c.args[0].src_name, c.args[0].rel_type, c.args[0].dst_name)
        for c in graph.add_relation.call_args_list
    }
    assert ("Project Alpha", "USES", "Qdrant") in rel_triples


def test_extractor_exception_does_not_propagate(tmp_path: Path):
    graph = MagicMock()
    graph.add_entity.side_effect = RuntimeError("graph exploded")
    graph.add_relation.side_effect = RuntimeError("graph exploded")
    attach_to_wiki(graph)

    wm = WikiManager(_cfg(tmp_path))
    # Must not raise even though the graph hook blows up on every call.
    page = wm.create_page(
        "Alpha", "Project Alpha uses Qdrant.", category="notes",
    )
    assert page.path is not None
    assert page.path.exists()
    # The hook was invoked — but the exception was swallowed.
    assert graph.add_entity.called


def test_extractor_top_level_exception_swallowed(tmp_path: Path):
    """If the extractor itself raises, WikiManager.create_page must still succeed."""
    broken = MagicMock(spec=EntityExtractor)
    broken.extract.side_effect = RuntimeError("extractor exploded")

    graph = MagicMock()
    attach_to_wiki(graph, extractor=broken)

    wm = WikiManager(_cfg(tmp_path))
    page = wm.create_page("Alpha", "Project Alpha uses Qdrant.")
    assert page.path is not None and page.path.exists()
    # Extractor was consulted but its explosion did not bubble up.
    assert broken.extract.called
    # And because extraction failed, no graph writes happened.
    graph.add_entity.assert_not_called()
    graph.add_relation.assert_not_called()


# ── Dataclass sanity ─────────────────────────────────────────────

def test_extracted_entity_defaults():
    ent = ExtractedEntity(label="Project", name="Alpha")
    assert ent.properties == {}


def test_attach_to_wiki_is_idempotent(tmp_path: Path):
    """Regression: Codex caught that repeated attach_to_wiki calls stacked
    duplicate closures because each call's closure has a different identity."""
    from rke.knowledge.extractor import attach_to_wiki, detach_from_wiki
    from rke.wiki import manager as wm_mod

    graph = MagicMock()
    attach_to_wiki(graph)
    after_first = len(wm_mod._post_create_hooks)
    attach_to_wiki(graph)
    after_second = len(wm_mod._post_create_hooks)
    attach_to_wiki(graph)
    after_third = len(wm_mod._post_create_hooks)

    assert after_first == after_second == after_third, (
        f"hook count drifted: {after_first} -> {after_second} -> {after_third}"
    )

    detach_from_wiki()
    assert len(wm_mod._post_create_hooks) == after_first - 1


def test_detach_from_wiki_when_not_attached_is_a_noop():
    from rke.knowledge.extractor import detach_from_wiki
    from rke.wiki import manager as wm_mod

    before = len(wm_mod._post_create_hooks)
    detach_from_wiki()
    detach_from_wiki()  # twice
    assert len(wm_mod._post_create_hooks) == before
