"""Chat-history adapter — LangChain-style ConversationBufferMemory.

Gives users migrating from LangChain a familiar interface while persisting
messages into the RKE wiki (and, optionally, indexing them via KnowledgeBase
for long-term semantic recall).

Design:
    * A "thread" is a single wiki page at category ``threads`` whose body is a
      sequence of ``### {timestamp} {role}`` blocks followed by the content.
    * Every mutation rewrites the page so a mid-run crash cannot lose data.
    * When the buffer grows past ``summarize_threshold`` the oldest
      ``len - buffer_size`` messages are summarised and rolled into a dedicated
      archive page at category ``threads/{thread_id}/archive``.

No dependency on langchain — this re-implements the subset we care about.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config import Config, load_config
from ..wiki.knowledge_base import CombinedHit, KnowledgeBase
from ..wiki.manager import WikiManager, slugify

_VALID_ROLES = {"user", "assistant", "system"}
# Matches a block header: "### 2024-01-01T00:00:00+00:00 user [optional JSON]".
# The trailing JSON is the message's metadata; absent for empty metadata.
_BLOCK_RE = re.compile(
    r"^###\s+(?P<ts>\S+)\s+(?P<role>user|assistant|system)"
    r"(?:\s+(?P<meta>\{.*\}))?\s*$",
    re.MULTILINE,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str  # ISO-8601 UTC
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Render as a single ``### ts role [json-metadata]`` block.

        Metadata is appended to the header as a single-line JSON object iff
        non-empty, so reload round-trips it. Empty metadata produces the
        bare ``### ts role`` form for cleaner output.
        """
        if self.metadata:
            try:
                meta_json = json.dumps(self.metadata, separators=(",", ":"), default=str)
                return f"### {self.timestamp} {self.role} {meta_json}\n{self.content}\n"
            except (TypeError, ValueError):
                pass  # fall through to no-metadata form
        return f"### {self.timestamp} {self.role}\n{self.content}\n"


def _render_messages(messages: list[Message]) -> str:
    return "\n".join(m.render() for m in messages)


def _parse_messages(body: str) -> list[Message]:
    """Round-trip counterpart to :func:`_render_messages`.

    Non-matching preamble is ignored; each ``### ts role`` header starts a new
    block whose body runs until the next header or end-of-text.
    """
    matches = list(_BLOCK_RE.finditer(body))
    out: list[Message] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip("\n")
        meta_str = m.group("meta")
        meta: dict[str, Any] = {}
        if meta_str:
            try:
                parsed = json.loads(meta_str)
                if isinstance(parsed, dict):
                    meta = parsed
            except (json.JSONDecodeError, TypeError):
                meta = {}
        out.append(Message(
            role=m.group("role"),
            content=content,
            timestamp=m.group("ts"),
            metadata=meta,
        ))
    return out


def _default_summarizer(messages: list[Message]) -> str:
    if not messages:
        return ""
    first_ts = messages[0].timestamp
    last_ts = messages[-1].timestamp
    header = f"[Summary of {len(messages)} messages from {first_ts} to {last_ts}]"
    parts = [m.content[:200] for m in messages]
    joined = "\n— ".join(parts)
    body = f"{header}\n— {joined}"
    return body[:2000]


class ChatMemory:
    """LangChain-compatible chat buffer backed by the RKE wiki."""

    def __init__(
        self,
        *,
        thread_id: str,
        kb: KnowledgeBase | None = None,
        config: Config | None = None,
        buffer_size: int = 50,
        summarize_threshold: int = 100,
        summarizer: Callable[[list[Message]], str] | None = None,
    ) -> None:
        if not thread_id:
            raise ValueError("thread_id must be non-empty")
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        if summarize_threshold <= buffer_size:
            raise ValueError("summarize_threshold must be > buffer_size")

        self.thread_id = thread_id
        self.slug = slugify(thread_id)
        self.buffer_size = buffer_size
        self.summarize_threshold = summarize_threshold
        self.summarizer = summarizer or _default_summarizer

        self.config = config or (kb.config if kb is not None else load_config())
        self.kb = kb
        # Use kb.wiki when available so both sides see the same root.
        self.wiki: WikiManager = kb.wiki if kb is not None else WikiManager(self.config)

        self._category = "threads"
        self._archive_category = f"threads/{self.slug}/archive"
        self._buffer: list[Message] = self._load()

    # ── Public API ───────────────────────────────────────────────
    def add_user_message(self, content: str, **metadata: Any) -> Message:
        return self.add_message("user", content, **metadata)

    def add_assistant_message(self, content: str, **metadata: Any) -> Message:
        return self.add_message("assistant", content, **metadata)

    def add_message(self, role: str, content: str, **metadata: Any) -> Message:
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {role!r}")
        msg = Message(
            role=role,
            content=content,
            timestamp=_utcnow_iso(),
            metadata=dict(metadata),
        )
        self._buffer.append(msg)
        self._persist()
        return msg

    def history(self, last_n: int | None = None) -> list[Message]:
        """Chronological (oldest first), matching LangChain semantics."""
        if last_n is None:
            return list(self._buffer)
        if last_n <= 0:
            return []
        return list(self._buffer[-last_n:])

    def clear(self) -> None:
        self._buffer = []
        self._persist()

    def to_prompt_string(self, last_n: int | None = None) -> str:
        msgs = self.history(last_n)
        return "".join(f"{m.role}: {m.content}\n" for m in msgs)

    def summarize_and_archive(self) -> str | None:
        if len(self._buffer) < self.summarize_threshold:
            return None

        split = len(self._buffer) - self.buffer_size
        if split <= 0:
            return None
        archived = self._buffer[:split]
        remaining = self._buffer[split:]
        summary = self.summarizer(archived)

        # Persist the archive page with raw messages + summary.
        archive_n = self._next_archive_n()
        archive_title = f"archive-{archive_n}"
        raw_body = _render_messages(archived)
        archive_body = (
            f"## Summary\n\n{summary}\n\n"
            f"## Raw messages ({len(archived)})\n\n{raw_body}"
        )
        self.wiki.create_page(
            archive_title,
            archive_body,
            category=self._archive_category,
            tags=["thread-archive", self.slug],
            overwrite=True,
        )

        # Prepend the summary as a system message and drop the archived raws.
        summary_msg = Message(
            role="system",
            content=summary,
            timestamp=_utcnow_iso(),
            metadata={"archive": archive_title, "archived_count": len(archived)},
        )
        self._buffer = [summary_msg, *remaining]
        self._persist()
        return summary

    def search_long_term(self, query: str, limit: int = 5) -> list[CombinedHit]:
        if self.kb is None:
            raise RuntimeError(
                "search_long_term requires a KnowledgeBase; pass kb= on construction",
            )
        return self.kb.query(query, wiki_limit=limit, vector_limit=limit)

    # ── Internal ─────────────────────────────────────────────────
    def _page_title(self) -> str:
        # Titles become slugs via slugify, so keep it aligned with self.slug.
        return self.thread_id

    def _load(self) -> list[Message]:
        page = self.wiki.get_page(self.slug, category=self._category)
        if page is None:
            return []
        return _parse_messages(page.body)

    def _persist(self) -> None:
        body = _render_messages(self._buffer) if self._buffer else ""
        self.wiki.create_page(
            self._page_title(),
            body,
            category=self._category,
            tags=["thread", self.slug],
            overwrite=True,
        )

    def _next_archive_n(self) -> int:
        existing = self.wiki.list_pages(category=self._archive_category)
        max_n = 0
        for p in existing:
            if p.slug.startswith("archive-"):
                try:
                    n = int(p.slug.split("-", 1)[1])
                except ValueError:
                    continue
                if n > max_n:
                    max_n = n
        return max_n + 1
