"""RLM environment — REPL-style operations over the knowledge base.

Implements the four core operations from Zhang & Khattab (2025):
    peek(slug)             → fetch first N lines of a wiki page
    grep(pattern, ...)     → regex over wiki + vector chunks
    partition(query, k)    → k-cluster the top hits to break a query into sub-questions
    sub_rlm(question)      → recursive call: run another RLM completion on the result

Agents never receive the full corpus; they call these operations to explore.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, load_config
from ..wiki.knowledge_base import KnowledgeBase

log = logging.getLogger(__name__)


@dataclass
class Trace:
    op: str
    args: dict[str, Any]
    result_summary: str


@dataclass
class Environment:
    """Stateful REPL handed to an RLM iteration."""

    kb: KnowledgeBase
    config: Config = field(default_factory=load_config)
    trace: list[Trace] = field(default_factory=list)
    max_peek_chars: int = 4_000

    # ── Operations ───────────────────────────────────────────────
    def peek(self, slug: str, lines: int = 40) -> str:
        page = self.kb.wiki.get_page(slug)
        if not page:
            self._log("peek", {"slug": slug}, "not found")
            return f"(no wiki page found for slug={slug!r})"
        head = "\n".join(page.body.splitlines()[:lines])[: self.max_peek_chars]
        self._log("peek", {"slug": slug, "lines": lines}, f"{len(head)} chars")
        return f"# {page.title}\n\n{head}"

    def grep(self, pattern: str, in_wiki: bool = True, in_vectors: bool = True, limit: int = 10) -> list[dict[str, Any]]:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return [{"error": f"invalid regex: {exc}"}]
        out: list[dict[str, Any]] = []
        if in_wiki:
            for page in self.kb.wiki.list_pages():
                for ln_no, line in enumerate(page.body.splitlines(), 1):
                    if regex.search(line):
                        out.append({
                            "source": "wiki",
                            "slug": page.slug,
                            "line": ln_no,
                            "text": line.strip()[:200],
                        })
                        if len(out) >= limit:
                            break
                if len(out) >= limit:
                    break
        if in_vectors and len(out) < limit:
            try:
                hits = self.kb.vectors.search(pattern, limit=limit - len(out))
                for h in hits:
                    if regex.search(h.text):
                        out.append({
                            "source": "vector",
                            "id": h.id,
                            "score": h.score,
                            "text": h.text[:200],
                        })
            except Exception as exc:
                out.append({"warning": f"vector grep skipped: {exc}"})
        self._log("grep", {"pattern": pattern, "limit": limit}, f"{len(out)} hits")
        return out

    def partition(self, query: str, k: int = 3) -> list[str]:
        """Split a query into k sub-queries using top wiki/vector hits.

        This is intentionally heuristic: take the top k combined hits and
        produce a sub-question from each title. A future version can use
        an LLM to do semantic clustering.
        """
        hits = self.kb.query(query, wiki_limit=k, vector_limit=k)[:k]
        if not hits:
            self._log("partition", {"query": query, "k": k}, "no hits")
            return [query]
        subs = [f"{query} — focus: {h.title}" for h in hits]
        self._log("partition", {"query": query, "k": k}, f"{len(subs)} sub-queries")
        return subs

    def sub_rlm(self, question: str, max_depth: int | None = None) -> str:
        """Recursive call. Imported lazily to avoid a cycle."""
        from .router import RLMRouter

        depth_cap = max_depth if max_depth is not None else int(
            self.config.rlm.get("max_depth", 5)
        )
        if depth_cap <= 0:
            self._log("sub_rlm", {"question": question}, "depth exhausted")
            return "(sub_rlm depth exhausted)"
        router = RLMRouter(config=self.config, kb=self.kb)
        result = router.complete(query=question, _depth=depth_cap - 1)
        self._log("sub_rlm", {"question": question}, f"{len(result.answer)} chars")
        return result.answer

    # ── Internals ────────────────────────────────────────────────
    def _log(self, op: str, args: dict[str, Any], summary: str) -> None:
        self.trace.append(Trace(op=op, args=args, result_summary=summary))


def make_environment(config: Config | None = None) -> Environment:
    cfg = config or load_config()
    return Environment(kb=KnowledgeBase(cfg), config=cfg)
