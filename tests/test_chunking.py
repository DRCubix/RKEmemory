"""Tests for the chunking helper."""

from rke.wiki.knowledge_base import chunk_text


def test_short_text_is_one_chunk():
    out = chunk_text("hello", chunk_size=1024, overlap=128)
    assert out == ["hello"]


def test_chunks_overlap_correctly():
    text = "ABCDEFGHIJ" * 50  # 500 chars
    out = chunk_text(text, chunk_size=100, overlap=20)
    assert len(out) >= 5
    # Each consecutive pair should share `overlap` chars at the boundary
    for a, b in zip(out, out[1:], strict=False):
        assert a[-20:] == b[:20] or a.endswith(b[:20])


def test_chunk_size_zero_returns_input():
    assert chunk_text("hello", chunk_size=0) == ["hello"]
