"""Tests for the Embedder.

Two layers of tests:
  1. Shape and dtype tests using a lightweight mock model -- no network
     access or real model download needed. These run instantly and cover
     the Embedder's own logic (batching, normalization, alignment).
  2. A real-model integration test (marked slow) that downloads
     all-MiniLM-L6-v2 and verifies semantic similarity actually works:
     similar sentences should be closer than dissimilar ones.

Run fast tests only (no download):
  python -m pytest tests/test_embedder.py -v -m "not slow"

Run all tests including real model:
  python -m pytest tests/test_embedder.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embedder import Embedder
from ingestion.chunker import Chunk


# ---------------------------------------------------------------------------
# Mock embedder for fast unit tests (no model download)
# ---------------------------------------------------------------------------

def _make_mock_embedder(dim: int = 384) -> Embedder:
    """Return an Embedder whose underlying model is replaced with a mock
    that returns random unit vectors. Lets us test shape, dtype, and
    alignment without any network access.
    """
    embedder = Embedder.__new__(Embedder)
    embedder.model_name = "mock"
    embedder.batch_size = 64
    embedder.normalize = True

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim

    def mock_encode(texts, **kwargs):
        rng = np.random.default_rng(seed=len(texts))
        vecs = rng.standard_normal((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock_model.encode.side_effect = mock_encode
    embedder._model = mock_model
    return embedder


# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------

def test_embed_returns_correct_shape():
    embedder = _make_mock_embedder(dim=384)
    texts = ["hello world", "foo bar", "baz qux"]
    result = embedder.embed(texts)
    assert result.shape == (3, 384), f"expected (3, 384), got {result.shape}"


def test_embed_returns_float32():
    embedder = _make_mock_embedder()
    result = embedder.embed(["test"])
    assert result.dtype == np.float32, f"expected float32, got {result.dtype}"


def test_embed_empty_list_returns_empty_array():
    embedder = _make_mock_embedder(dim=384)
    result = embedder.embed([])
    assert result.shape == (0, 384)


def test_embed_single_text_returns_2d_array():
    embedder = _make_mock_embedder()
    result = embedder.embed(["just one sentence"])
    assert result.ndim == 2
    assert result.shape[0] == 1


def test_embed_normalized_vectors_have_unit_norm():
    embedder = _make_mock_embedder(dim=384)
    result = embedder.embed(["hello", "world", "foo"])
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5,
        err_msg="Embeddings should be unit vectors after normalization")


def test_embed_chunks_aligns_with_chunk_list():
    embedder = _make_mock_embedder(dim=384)
    chunks = [
        Chunk(text=f"chunk text {i}", source="test", chunk_index=i,
              start_char=i*10, end_char=i*10+9)
        for i in range(5)
    ]
    result = embedder.embed_chunks(chunks)
    assert result.shape == (5, 384), (
        f"embed_chunks should return one vector per chunk, got {result.shape}"
    )


def test_embedding_dim_property():
    embedder = _make_mock_embedder(dim=128)
    assert embedder.embedding_dim == 128


def test_different_texts_produce_different_embeddings():
    embedder = _make_mock_embedder(dim=384)
    result = embedder.embed(["hello world", "completely different text"])
    # With a random mock, two different texts should produce different vectors.
    assert not np.allclose(result[0], result[1]), (
        "Different texts should produce different embeddings"
    )


# ---------------------------------------------------------------------------
# Real model integration test (requires network + sentence-transformers)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_model_semantic_similarity():
    """Verify that the real all-MiniLM-L6-v2 model produces semantically
    meaningful embeddings: similar sentences should have higher cosine
    similarity than dissimilar ones.

    This test downloads ~90MB on first run. Skip with -m "not slow".
    """
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    similar_a = "The cat sat on the mat"
    similar_b = "A cat is sitting on a mat"
    dissimilar = "The stock market fell sharply today"

    vecs = embedder.embed([similar_a, similar_b, dissimilar])
    sim_similar = float(np.dot(vecs[0], vecs[1]))
    sim_dissimilar = float(np.dot(vecs[0], vecs[2]))

    assert sim_similar > sim_dissimilar, (
        f"Similar sentences (sim={sim_similar:.3f}) should have higher "
        f"cosine similarity than dissimilar ones (sim={sim_dissimilar:.3f})"
    )


@pytest.mark.slow
def test_real_model_output_shape():
    embedder = Embedder()
    result = embedder.embed(["hello", "world"])
    assert result.shape == (2, 384)
    assert result.dtype == np.float32
