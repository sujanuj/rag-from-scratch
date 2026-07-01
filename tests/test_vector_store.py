"""Tests for the flat vector store.

Four things verified:
  1. Add and size tracking.
  2. Search returns correct top-K results in the right order.
  3. Exact retrieval: a query identical to a stored vector must be the
     top result with similarity ~1.0.
  4. Save/load round-trip: a saved and reloaded store produces identical
     search results to the original.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunker import Chunk
from retriever.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(i: int) -> Chunk:
    return Chunk(
        text=f"chunk text {i}",
        source="test.txt",
        chunk_index=i,
        start_char=i * 20,
        end_char=i * 20 + 19,
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _random_unit(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return _unit(v)


# ---------------------------------------------------------------------------
# Add and size
# ---------------------------------------------------------------------------

def test_store_starts_empty():
    store = VectorStore(dim=8)
    assert store.size == 0


def test_add_increases_size():
    store = VectorStore(dim=8)
    chunks = [_make_chunk(i) for i in range(5)]
    embeddings = np.stack([_random_unit(8, i) for i in range(5)])
    store.add(chunks, embeddings)
    assert store.size == 5


def test_add_multiple_batches_accumulates():
    store = VectorStore(dim=8)
    for batch in range(3):
        chunks = [_make_chunk(batch * 4 + i) for i in range(4)]
        embeddings = np.stack([_random_unit(8, batch * 4 + i) for i in range(4)])
        store.add(chunks, embeddings)
    assert store.size == 12


def test_add_empty_batch_is_safe():
    store = VectorStore(dim=8)
    store.add([], np.zeros((0, 8), dtype=np.float32))
    assert store.size == 0


def test_add_mismatched_lengths_raises():
    store = VectorStore(dim=8)
    chunks = [_make_chunk(0), _make_chunk(1)]
    embeddings = np.stack([_random_unit(8, 0)])  # only 1 embedding for 2 chunks
    with pytest.raises(ValueError):
        store.add(chunks, embeddings)


def test_add_wrong_dim_raises():
    store = VectorStore(dim=8)
    chunks = [_make_chunk(0)]
    embeddings = np.stack([_random_unit(16, 0)])  # dim=16 != store.dim=8
    with pytest.raises(ValueError):
        store.add(chunks, embeddings)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_search_empty_store_raises():
    store = VectorStore(dim=8)
    query = _random_unit(8, 99)
    with pytest.raises(ValueError):
        store.search(query)


def test_search_returns_top_k_results():
    store = VectorStore(dim=8)
    chunks = [_make_chunk(i) for i in range(10)]
    embeddings = np.stack([_random_unit(8, i) for i in range(10)])
    store.add(chunks, embeddings)

    results = store.search(_random_unit(8, 42), top_k=3)
    assert len(results) == 3


def test_search_results_sorted_by_descending_score():
    store = VectorStore(dim=8)
    chunks = [_make_chunk(i) for i in range(10)]
    embeddings = np.stack([_random_unit(8, i) for i in range(10)])
    store.add(chunks, embeddings)

    results = store.search(_random_unit(8, 42), top_k=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), (
        f"Results not sorted by descending score: {scores}"
    )


def test_search_exact_match_is_top_result():
    # A query identical to a stored vector must be the top result
    # with similarity close to 1.0.
    store = VectorStore(dim=32)
    target_vec = _random_unit(32, 7)
    chunks = [_make_chunk(i) for i in range(20)]
    embeddings = np.stack([_random_unit(32, i) for i in range(20)])
    # Replace chunk 7's embedding with target_vec
    embeddings[7] = target_vec
    store.add(chunks, embeddings)

    results = store.search(target_vec, top_k=1)
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_index == 7, (
        f"Expected chunk 7 as top result, got chunk {top_chunk.chunk_index}"
    )
    assert top_score > 0.999, f"Exact match should have similarity ~1.0, got {top_score}"


def test_search_top_k_clamped_to_store_size():
    store = VectorStore(dim=8)
    chunks = [_make_chunk(i) for i in range(3)]
    embeddings = np.stack([_random_unit(8, i) for i in range(3)])
    store.add(chunks, embeddings)

    results = store.search(_random_unit(8, 0), top_k=100)
    assert len(results) == 3  # clamped to store size


def test_search_wrong_query_dim_raises():
    store = VectorStore(dim=8)
    store.add([_make_chunk(0)], np.stack([_random_unit(8, 0)]))
    with pytest.raises(ValueError):
        store.search(_random_unit(16, 0))  # wrong dim


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_search_results():
    store = VectorStore(dim=32)
    chunks = [_make_chunk(i) for i in range(20)]
    embeddings = np.stack([_random_unit(32, i) for i in range(20)])
    store.add(chunks, embeddings)

    query = _random_unit(32, 99)
    original_results = store.search(query, top_k=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "store")
        store.save(path)
        loaded = VectorStore.load(path)

    loaded_results = loaded.search(query, top_k=5)

    assert len(original_results) == len(loaded_results)
    for (orig_chunk, orig_score), (load_chunk, load_score) in zip(
        original_results, loaded_results
    ):
        assert orig_chunk.chunk_index == load_chunk.chunk_index
        assert abs(orig_score - load_score) < 1e-5


def test_save_load_preserves_chunk_metadata():
    store = VectorStore(dim=8)
    chunk = Chunk(text="hello world", source="doc.txt",
                  chunk_index=3, start_char=10, end_char=21)
    store.add([chunk], np.stack([_random_unit(8, 0)]))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "store")
        store.save(path)
        loaded = VectorStore.load(path)

    assert loaded.size == 1
    results = loaded.search(_random_unit(8, 0), top_k=1)
    loaded_chunk = results[0][0]
    assert loaded_chunk.text == "hello world"
    assert loaded_chunk.source == "doc.txt"
    assert loaded_chunk.chunk_index == 3
    assert loaded_chunk.start_char == 10
    assert loaded_chunk.end_char == 21
