"""Tests for the ChatMemory LangChain-compat adapter.

No Qdrant / FalkorDB required: KnowledgeBase is bypassed and only the wiki
side is exercised. ``search_long_term`` is covered with a Mock.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rke.adapters.chat_memory import (
    ChatMemory,
    Message,
    _default_summarizer,
    _parse_messages,
)
from rke.config import DEFAULTS, Config
from rke.wiki.knowledge_base import CombinedHit


def _cfg(tmp_path: Path) -> Config:
    raw = {**DEFAULTS, "wiki": {"path": str(tmp_path / "wiki")}}
    return Config(raw=raw)


def _iso_parseable(ts: str) -> bool:
    try:
        datetime.fromisoformat(ts)
        return True
    except ValueError:
        return False


def test_add_user_message_returns_message_with_parseable_ts(tmp_path: Path):
    mem = ChatMemory(thread_id="t1", config=_cfg(tmp_path))
    msg = mem.add_user_message("hello there", source="cli")

    assert isinstance(msg, Message)
    assert msg.role == "user"
    assert msg.content == "hello there"
    assert _iso_parseable(msg.timestamp)
    assert msg.metadata == {"source": "cli"}


def test_add_assistant_message_role_and_ts(tmp_path: Path):
    mem = ChatMemory(thread_id="t1", config=_cfg(tmp_path))
    msg = mem.add_assistant_message("hi!")

    assert msg.role == "assistant"
    assert msg.content == "hi!"
    assert _iso_parseable(msg.timestamp)


def test_history_is_chronological(tmp_path: Path):
    mem = ChatMemory(thread_id="t1", config=_cfg(tmp_path))
    mem.add_user_message("one")
    mem.add_assistant_message("two")
    mem.add_user_message("three")

    hist = mem.history()
    assert [m.content for m in hist] == ["one", "two", "three"]

    assert [m.content for m in mem.history(last_n=2)] == ["two", "three"]
    assert mem.history(last_n=0) == []


def test_to_prompt_string_format(tmp_path: Path):
    mem = ChatMemory(thread_id="t1", config=_cfg(tmp_path))
    mem.add_user_message("hi")
    mem.add_assistant_message("hello back")

    out = mem.to_prompt_string()
    assert out == "user: hi\nassistant: hello back\n"


def test_clear_empties_buffer_and_wiki_body(tmp_path: Path):
    cfg = _cfg(tmp_path)
    mem = ChatMemory(thread_id="t1", config=cfg)
    mem.add_user_message("doomed")
    mem.add_assistant_message("also doomed")

    mem.clear()
    assert mem.history() == []

    # And a fresh instance sees no messages either.
    mem2 = ChatMemory(thread_id="t1", config=cfg)
    assert mem2.history() == []

    # Page still exists but body has no message blocks.
    page = mem.wiki.get_page("t1", category="threads")
    assert page is not None
    assert "###" not in page.body


def test_persistence_round_trip(tmp_path: Path):
    cfg = _cfg(tmp_path)
    first = ChatMemory(thread_id="my-thread", config=cfg)
    first.add_user_message("remember me")
    first.add_assistant_message("sure will")
    first.add_user_message("and this line\nhas two")

    second = ChatMemory(thread_id="my-thread", config=cfg)
    reloaded = second.history()
    assert [m.role for m in reloaded] == ["user", "assistant", "user"]
    assert [m.content for m in reloaded] == [
        "remember me",
        "sure will",
        "and this line\nhas two",
    ]
    for m in reloaded:
        assert _iso_parseable(m.timestamp)


def test_summarize_and_archive_triggers_above_threshold(tmp_path: Path):
    cfg = _cfg(tmp_path)
    mem = ChatMemory(
        thread_id="chatty",
        config=cfg,
        buffer_size=3,
        summarize_threshold=6,
    )
    for i in range(6):
        mem.add_user_message(f"msg {i}")

    summary = mem.summarize_and_archive()
    assert summary is not None
    assert summary.startswith("[Summary of 3 messages")

    # Buffer is now: [system summary, msg 3, msg 4, msg 5]
    hist = mem.history()
    assert len(hist) == 4
    assert hist[0].role == "system"
    assert [m.content for m in hist[1:]] == ["msg 3", "msg 4", "msg 5"]

    # Archive page exists with raw messages preserved.
    archive_page = mem.wiki.get_page("archive-1", category="threads/chatty/archive")
    assert archive_page is not None
    assert "msg 0" in archive_page.body
    assert "msg 2" in archive_page.body
    assert "## Summary" in archive_page.body


def test_summarize_and_archive_noop_below_threshold(tmp_path: Path):
    mem = ChatMemory(
        thread_id="quiet",
        config=_cfg(tmp_path),
        buffer_size=3,
        summarize_threshold=6,
    )
    for i in range(5):
        mem.add_user_message(f"msg {i}")

    result = mem.summarize_and_archive()
    assert result is None
    assert len(mem.history()) == 5
    # No archive category page created.
    assert mem.wiki.get_page("archive-1", category="threads/quiet/archive") is None


def test_custom_summarizer_is_invoked(tmp_path: Path):
    calls: list[list[Message]] = []

    def fake_summarizer(msgs: list[Message]) -> str:
        calls.append(list(msgs))
        return f"fake-summary-of-{len(msgs)}"

    mem = ChatMemory(
        thread_id="custom",
        config=_cfg(tmp_path),
        buffer_size=2,
        summarize_threshold=5,
        summarizer=fake_summarizer,
    )
    for i in range(5):
        mem.add_user_message(f"m{i}")

    summary = mem.summarize_and_archive()
    assert summary == "fake-summary-of-3"
    assert len(calls) == 1
    assert [m.content for m in calls[0]] == ["m0", "m1", "m2"]
    # First message in the buffer is the system summary from our callable.
    assert mem.history()[0].content == "fake-summary-of-3"


def test_search_long_term_delegates_to_kb(tmp_path: Path):
    from rke.wiki.manager import WikiManager

    kb = MagicMock()
    kb.config = _cfg(tmp_path)
    # Let wiki be a real WikiManager sharing the same root.
    kb.wiki = WikiManager(kb.config)
    expected = [CombinedHit(source="wiki", score=1.0, title="x", snippet="y")]
    kb.query.return_value = expected

    mem = ChatMemory(thread_id="kb-thread", kb=kb)
    got = mem.search_long_term("what did we discuss?", limit=4)

    kb.query.assert_called_once_with("what did we discuss?", wiki_limit=4, vector_limit=4)
    assert got is expected


def test_search_long_term_without_kb_raises(tmp_path: Path):
    mem = ChatMemory(thread_id="solo", config=_cfg(tmp_path))
    with pytest.raises(RuntimeError):
        mem.search_long_term("anything")


def test_parse_messages_roundtrip():
    msgs = [
        Message(role="user", content="hi", timestamp="2024-01-01T00:00:00+00:00"),
        Message(role="assistant", content="multi\nline\nbody", timestamp="2024-01-01T00:00:01+00:00"),
    ]
    body = "\n".join(m.render() for m in msgs)
    parsed = _parse_messages(body)
    assert [p.role for p in parsed] == ["user", "assistant"]
    assert [p.content for p in parsed] == ["hi", "multi\nline\nbody"]


def test_default_summarizer_truncates():
    long_content = "x" * 500
    msgs = [
        Message(role="user", content=long_content, timestamp="2024-01-01T00:00:00+00:00")
        for _ in range(50)
    ]
    out = _default_summarizer(msgs)
    assert len(out) <= 2000
    assert out.startswith("[Summary of 50 messages")
