"""Tests for the document chunker.

Four things verified:
  1. Basic chunking: correct chunk count, size, and ordering.
  2. Overlap: words from the end of chunk N appear at the start of chunk N+1.
  3. Character offsets: start_char/end_char correctly locate each chunk
     in the original text.
  4. Edge cases: empty text, single chunk, overlap validation.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunker import chunk_text, chunk_file, chunk_documents, Chunk


def _make_text(num_words: int) -> str:
    """Make a text with num_words distinct words for easy counting."""
    return " ".join(f"word{i}" for i in range(num_words))


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------

def test_single_chunk_when_text_fits():
    text = _make_text(10)
    chunks = chunk_text(text, source="test", chunk_size=20, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].source == "test"


def test_chunk_count_matches_expected():
    # 100 words, chunk_size=20, overlap=5 -> step=15
    # chunks cover: 0-19, 15-34, 30-49, 45-64, 60-79, 75-94, 90-99
    # = ceil((100 - 20) / 15) + 1 = 7
    text = _make_text(100)
    chunks = chunk_text(text, source="test", chunk_size=20, overlap=5)
    assert len(chunks) == 7


def test_chunk_indices_are_sequential():
    text = _make_text(60)
    chunks = chunk_text(text, source="test", chunk_size=20, overlap=4)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_all_words_covered():
    # Every word in the original text must appear in at least one chunk.
    text = _make_text(50)
    chunks = chunk_text(text, source="test", chunk_size=15, overlap=3)
    covered = set()
    for chunk in chunks:
        covered.update(chunk.text.split())
    assert covered == set(text.split())


def test_last_chunk_contains_last_word():
    text = _make_text(25)
    chunks = chunk_text(text, source="test", chunk_size=10, overlap=2)
    last_word = text.split()[-1]
    assert last_word in chunks[-1].text


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def test_overlap_words_appear_in_consecutive_chunks():
    # With chunk_size=10 and overlap=3, the last 3 words of chunk 0
    # must be the first 3 words of chunk 1.
    text = _make_text(30)
    chunks = chunk_text(text, source="test", chunk_size=10, overlap=3)
    assert len(chunks) >= 2

    tail_words = chunks[0].text.split()[-3:]
    head_words = chunks[1].text.split()[:3]
    assert tail_words == head_words, (
        f"Expected overlap words {tail_words} at start of chunk 1, got {head_words}"
    )


def test_overlap_zero_produces_no_repeated_words():
    text = _make_text(40)
    chunks = chunk_text(text, source="test", chunk_size=10, overlap=0)
    # No word should appear in more than one chunk.
    seen = {}
    for chunk in chunks:
        for word in chunk.text.split():
            assert word not in seen, f"{word} appeared in multiple chunks with overlap=0"
            seen[word] = chunk.chunk_index


# ---------------------------------------------------------------------------
# Character offsets
# ---------------------------------------------------------------------------

def test_start_char_end_char_locate_chunk_in_original():
    text = "The quick brown fox jumps over the lazy dog near the river bank today"
    chunks = chunk_text(text, source="test", chunk_size=5, overlap=1)
    for chunk in chunks:
        extracted = text[chunk.start_char:chunk.end_char]
        assert extracted == chunk.text, (
            f"text[{chunk.start_char}:{chunk.end_char}] = {extracted!r} "
            f"!= chunk.text = {chunk.text!r}"
        )


def test_first_chunk_starts_at_zero():
    text = "hello world foo bar"
    chunks = chunk_text(text, source="test", chunk_size=4, overlap=1)
    assert chunks[0].start_char == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_text_returns_no_chunks():
    assert chunk_text("", source="test") == []
    assert chunk_text("   \n  ", source="test") == []


def test_overlap_equal_to_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello world", source="test", chunk_size=5, overlap=5)


def test_overlap_greater_than_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello world", source="test", chunk_size=5, overlap=6)


def test_chunk_documents_multiple_sources():
    docs = [
        {"text": _make_text(30), "source": "doc_a"},
        {"text": _make_text(30), "source": "doc_b"},
    ]
    chunks = chunk_documents(docs, chunk_size=15, overlap=2)
    sources = {c.source for c in chunks}
    assert "doc_a" in sources
    assert "doc_b" in sources


def test_chunk_file(tmp_path):
    content = _make_text(40)
    f = tmp_path / "sample.txt"
    f.write_text(content)
    chunks = chunk_file(f, chunk_size=15, overlap=3)
    assert len(chunks) > 0
    assert chunks[0].source == "sample.txt"
    assert all(c.source == "sample.txt" for c in chunks)
