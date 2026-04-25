"""Agent integration — produce a context block ready for an LLM agent prompt.

Agents do not see the full corpus. They get:
  • a synthesized summary of top wiki/vector hits
  • optional graph subgraph
  • role-specific framing for Claude / Codex / Gemini / generic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import Config, load_config
from .graph_store import GraphStore
from .wiki.knowledge_base import CombinedHit, KnowledgeBase

log = logging.getLogger(__name__)


@dataclass
class GatheredContext:
    query: str
    hits: list[CombinedHit] = field(default_factory=list)
    graph: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.hits and not self.graph.get("subgraph")


def gather_context(
    query: str,
    *,
    config: Config | None = None,
    kb: KnowledgeBase | None = None,
    graph: GraphStore | None = None,
    use_graph: bool = True,
    wiki_limit: int = 3,
    vector_limit: int = 5,
) -> GatheredContext:
    cfg = config or load_config()
    kb = kb or KnowledgeBase(cfg)
    hits = kb.query(query, wiki_limit=wiki_limit, vector_limit=vector_limit)

    graph_payload: dict[str, Any] = {}
    if use_graph:
        gs = graph or GraphStore(cfg)
        try:
            if gs.ping():
                graph_payload = gs.graphrag_query(query)
        except Exception as exc:
            log.debug("graph context skipped: %s", exc)

    return GatheredContext(query=query, hits=hits, graph=graph_payload)


def format_context_for_agent(ctx: GatheredContext, agent: str = "generic") -> str:
    """Produce a prompt-ready context block."""
    if ctx.is_empty():
        return f"<context>\n  (no knowledge found for: {ctx.query})\n</context>"

    lines: list[str] = []
    lines.append("<context>")
    if ctx.hits:
        lines.append(f"  <evidence count=\"{len(ctx.hits)}\">")
        for i, h in enumerate(ctx.hits, 1):
            lines.append(
                f"    [{i}] ({h.source}, score={h.score:.2f}) {h.title}\n"
                f"        {h.snippet.replace(chr(10), ' ')[:280]}"
            )
        lines.append("  </evidence>")

    sub = ctx.graph.get("subgraph") if ctx.graph else None
    if sub:
        lines.append("  <graph>")
        for node in sub[:5]:
            ns = ", ".join(node.get("neighbors") or [])[:200]
            lines.append(
                f"    • {node.get('root')} {node.get('labels')} → [{ns}]"
            )
        lines.append("  </graph>")
    lines.append("</context>")

    body = "\n".join(lines)
    role = _role_prefix(agent)
    return f"{role}\n\n{body}"


def _role_prefix(agent: str) -> str:
    a = (agent or "").lower()
    if a == "claude":
        return (
            "You are Claude. Use the <context> below as authoritative project memory. "
            "Cite sources by [N] when you draw on them; if context is empty, say so."
        )
    if a == "codex":
        return (
            "You are Codex / a code-focused assistant. Treat <context> as project memory; "
            "prefer code snippets and file paths from it over guessing."
        )
    if a == "gemini":
        return (
            "You are Gemini. The <context> below is curated project memory. "
            "Ground your reasoning in it; mark anything you infer beyond it."
        )
    return (
        "Treat the <context> below as authoritative project memory. "
        "Reference it by [N] and stay grounded."
    )
