"""Entity/relation extraction from free text.

Mem0/Zep-style auto-extraction. Three backends (regex default, anthropic,
openai). LLM backends return JSON and gracefully fall back to regex on
parse failure. :func:`attach_to_wiki` wires this into WikiManager so new
pages auto-feed the FalkorDB graph without blocking page creation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..graph_store import Entity, GraphStore, Relation
from ..wiki.manager import WikiPage, register_post_create

log = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    label: str  # "Person", "Project", "API", "Concept"
    name: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    src_name: str
    rel_type: str  # uppercase, snake_case
    dst_name: str
    properties: dict[str, Any] = field(default_factory=dict)


# ── Regex backend ────────────────────────────────────────────────

_MAX_ENTITIES = 30
_MAX_RELATIONS = 30

# Proper-noun-looking runs of 1-3 capitalised words. Plain-space (not \s) is
# deliberate — cross-newline matches produce junk like "Notes\n\nProject Alpha".
_NOUN_BODY = r"[A-Z][a-zA-Z0-9]+(?: [A-Z][a-zA-Z0-9]+){0,2}"
_NOUN_RE = re.compile(_NOUN_BODY)

# Common English sentence-starts / bareword junk. Filtered only as a
# single-word candidate; "Project Alpha" is fine, bare "Project" is not.
_STOPWORDS: frozenset[str] = frozenset({
    "The", "A", "An", "This", "That", "These", "Those",
    "When", "Where", "Why", "How", "What", "Who", "Which",
    "If", "Then", "Else", "And", "Or", "But", "So", "Because",
    "It", "They", "We", "You", "He", "She", "I",
    "Is", "Are", "Was", "Were", "Be", "Been", "Being",
    "Do", "Does", "Did", "Has", "Have", "Had",
    "Project", "Thing", "Stuff", "Note", "Notes",
    "Today", "Tomorrow", "Yesterday", "Yes", "No", "Ok", "Okay",
})


def _rel(verb: str) -> re.Pattern[str]:
    return re.compile(rf"({_NOUN_BODY})\s+{verb}\s+({_NOUN_BODY})")


# Relation patterns. Order matters — longer phrases must win over shorter
# substrings (e.g. "depends on" before "uses").
_REL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("DEPENDS_ON", _rel(r"depends\s+on")),
    ("OWNED_BY", _rel(r"owned\s+by")),
    ("USES", _rel(r"uses?")),
    ("EXTENDS", _rel(r"extends")),
    ("IMPLEMENTS", _rel(r"implements")),
    ("OWNS", _rel(r"owns")),
]


def _is_junk(candidate: str) -> bool:
    """Reject obvious non-entity strings.

    Bareword stopwords ("The", "Project" alone) are filtered. Compound
    entities that merely START with a stopword ("Project Alpha") pass.
    """
    if len(candidate) < 2:
        return True
    words = candidate.split(" ")
    # Single-letter bareword ("A", "I")
    if len(words) == 1 and len(words[0]) == 1:
        return True
    # Bareword stopword
    if len(words) == 1 and words[0] in _STOPWORDS:
        return True
    return False


def _regex_extract(
    text: str,
    *,
    hint_labels: list[str] | None = None,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    label = (hint_labels[0] if hint_labels else "Concept")
    relations: list[ExtractedRelation] = []
    seen_rel: set[tuple[str, str, str]] = set()
    for rel_type, pattern in _REL_PATTERNS:
        for m in pattern.finditer(text):
            src, dst = m.group(1).strip(), m.group(2).strip()
            key = (src, rel_type, dst)
            if _is_junk(src) or _is_junk(dst) or key in seen_rel:
                continue
            seen_rel.add(key)
            relations.append(ExtractedRelation(src_name=src, rel_type=rel_type, dst_name=dst))
            if len(relations) >= _MAX_RELATIONS:
                break
        if len(relations) >= _MAX_RELATIONS:
            break

    entities: list[ExtractedEntity] = []
    seen: set[str] = set()
    candidates: list[str] = []
    for rel in relations:
        candidates.extend([rel.src_name, rel.dst_name])
    candidates.extend(m.group(0).strip() for m in _NOUN_RE.finditer(text))
    for name in candidates:
        if name in seen or _is_junk(name):
            continue
        seen.add(name)
        entities.append(ExtractedEntity(label=label, name=name))
        if len(entities) >= _MAX_ENTITIES:
            break
    return entities, relations


# ── LLM backends ─────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = (
    "You extract knowledge-graph entities and relations from text. "
    "Reply with ONLY a JSON object of the form "
    '{"entities":[{"label":"Person|Project|API|Concept","name":"..."}],'
    '"relations":[{"src":"...","type":"UPPER_SNAKE","dst":"..."}]}. '
    "Relation types must be UPPERCASE snake_case. No prose, no markdown."
)

_DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini"}


def _parse_llm_json(raw: str) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """Best-effort JSON parse; raises on failure."""
    trimmed = raw.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```[a-zA-Z]*\s*", "", trimmed)
        trimmed = re.sub(r"\s*```$", "", trimmed)
    data = json.loads(trimmed)
    entities = [
        ExtractedEntity(
            label=str(e.get("label") or "Concept"),
            name=str(e["name"]),
            properties=dict(e.get("properties") or {}),
        )
        for e in (data.get("entities") or []) if e.get("name")
    ][:_MAX_ENTITIES]
    relations = [
        ExtractedRelation(
            src_name=str(r["src"]),
            rel_type=str(r.get("type") or "RELATED_TO").upper(),
            dst_name=str(r["dst"]),
            properties=dict(r.get("properties") or {}),
        )
        for r in (data.get("relations") or []) if r.get("src") and r.get("dst")
    ][:_MAX_RELATIONS]
    return entities, relations


def _llm_extract(provider: str, text: str, model: str) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    if provider == "anthropic":
        import anthropic  # type: ignore
        resp = anthropic.Anthropic().messages.create(
            model=model, max_tokens=1024, system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    else:  # openai
        import openai  # type: ignore
        resp = openai.OpenAI().chat.completions.create(
            model=model, max_tokens=1024,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        raw = resp.choices[0].message.content or ""
    return _parse_llm_json(raw)


# ── Public extractor class ───────────────────────────────────────

class EntityExtractor:
    """Pluggable entity/relation extractor.

    Three backends:
      - regex (default, no API needed) — pulls capitalised proper-noun-looking
        runs and pattern-matched relations like 'X uses Y', 'X depends on Y'.
        Good enough for tests and offline ops.
      - anthropic (requires ANTHROPIC_API_KEY; provider="anthropic")
      - openai (requires OPENAI_API_KEY; provider="openai")

    Public API:
      extract(text: str, *, hint_labels: list[str] | None = None)
        -> tuple[list[ExtractedEntity], list[ExtractedRelation]]
    """

    def __init__(self, provider: str = "regex", model: str | None = None) -> None:
        self.provider = (provider or "regex").lower().strip()
        self.model = (
            model
            or os.getenv("RKE_LLM_MODEL")
            or _DEFAULT_MODELS.get(self.provider, "")
        )

    def extract(
        self,
        text: str,
        *,
        hint_labels: list[str] | None = None,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        if not text or not text.strip():
            return [], []
        if self.provider in ("anthropic", "openai"):
            try:
                return _llm_extract(self.provider, text, self.model)
            except Exception as exc:
                log.warning("%s extract failed (%s); falling back to regex", self.provider, exc)
        return _regex_extract(text, hint_labels=hint_labels)


# ── Wiki integration ─────────────────────────────────────────────

def _write_to_graph(graph: GraphStore, page: WikiPage, entities, relations) -> None:
    labels: dict[str, str] = {}
    for ent in entities:
        try:
            graph.add_entity(Entity(
                label=ent.label, name=ent.name,
                properties={**ent.properties, "source_slug": page.slug},
            ))
            labels[ent.name] = ent.label
        except Exception as exc:
            log.warning("graph.add_entity(%r) failed: %s", ent.name, exc)
    for rel in relations:
        try:
            graph.add_relation(Relation(
                src_label=labels.get(rel.src_name, "Concept"),
                src_name=rel.src_name,
                rel_type=rel.rel_type,
                dst_label=labels.get(rel.dst_name, "Concept"),
                dst_name=rel.dst_name,
                properties={**rel.properties, "source_slug": page.slug},
            ))
        except Exception as exc:
            log.warning("graph.add_relation(%r) failed: %s", rel.rel_type, exc)


def attach_to_wiki(
    graph: GraphStore,
    *,
    extractor: EntityExtractor | None = None,
) -> None:
    """Register a post_create wiki hook that auto-feeds the graph.

    Any failure (extractor error, graph write error, etc.) is logged but
    NEVER raised — wiki page creation must not depend on graph availability.
    """
    ex = extractor or EntityExtractor()

    def _hook(page: WikiPage) -> None:
        try:
            entities, relations = ex.extract(f"{page.title}\n\n{page.body}")
            _write_to_graph(graph, page, entities, relations)
        except Exception as exc:
            log.warning("entity extraction hook failed for %r: %s", page.slug, exc)

    register_post_create(_hook)
