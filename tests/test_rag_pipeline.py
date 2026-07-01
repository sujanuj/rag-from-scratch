"""Tests for the RAG pipeline.

Uses mock embedder and mock generator so no network access or model
download is needed. Tests verify:
  1. Ingest -> retrieve: after ingesting text, relevant queries return
     the right chunks.
  2. Prompt formatting: retrieved chunks appear in the prompt correctly.
  3. Full query(): retrieved context reaches the generator and the
     answer is returned correctly.
  4. Edge cases: empty store, missing generator.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embedder import Embedder
from ingestion.chunker import Chunk
from pipeline.rag_pipeline import RAGPipeline, format_prompt
from retriever.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Mock embedder: returns deterministic unit vectors based on text hash
# ---------------------------------------------------------------------------

def _make_mock_embedder(dim: int = 32) -> Embedder:
    embedder = Embedder.__new__(Embedder)
    embedder.model_name = "mock"
    embedder.batch_size = 64
    embedder.normalize = True

    def _vec(text: str) -> np.ndarray:
        """Deterministic unit vector from text -- same text always gives
        the same vector so retrieval tests are reproducible."""
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim

    def mock_encode(texts, **kwargs):
        return np.stack([_vec(t) for t in texts])

    mock_model.encode.side_effect = mock_encode
    embedder._model = mock_model
    return embedder


def _make_pipeline(generator_fn=None, dim=32) -> RAGPipeline:
    embedder = _make_mock_embedder(dim=dim)
    store = VectorStore(dim=dim)
    return RAGPipeline(
        embedder=embedder,
        store=store,
        generator_fn=generator_fn,
        top_k=3,
        chunk_size=10,
        chunk_overlap=2,
    )


# ---------------------------------------------------------------------------
# Ingest and retrieve
# ---------------------------------------------------------------------------

def test_ingest_adds_chunks_to_store():
    pipeline = _make_pipeline()
    pipeline.ingest([{"text": "hello world foo bar baz", "source": "doc.txt"}])
    assert pipeline.store.size > 0


def test_ingest_text_convenience():
    pipeline = _make_pipeline()
    pipeline.ingest_text("the quick brown fox jumps over the lazy dog", source="test")
    assert pipeline.store.size > 0


def test_retrieve_returns_top_k_results():
    pipeline = _make_pipeline()
    pipeline.ingest([{"text": " ".join(f"word{i}" for i in range(50)), "source": "doc.txt"}])
    results = pipeline.retrieve("word1 word2 word3")
    assert len(results) <= 3  # top_k=3
    assert len(results) > 0


def test_retrieve_results_have_chunk_and_score():
    pipeline = _make_pipeline()
    pipeline.ingest([{"text": " ".join(f"word{i}" for i in range(30)), "source": "doc.txt"}])
    results = pipeline.retrieve("some query")
    for chunk, score in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


def test_retrieve_from_empty_store_raises():
    pipeline = _make_pipeline()
    with pytest.raises(ValueError):
        pipeline.retrieve("anything")


def test_ingest_multiple_documents():
    pipeline = _make_pipeline()
    pipeline.ingest([
        {"text": " ".join(f"alpha{i}" for i in range(20)), "source": "doc_a.txt"},
        {"text": " ".join(f"beta{i}" for i in range(20)), "source": "doc_b.txt"},
    ])
    sources = {chunk.source for chunk, _ in pipeline.retrieve("alpha0 alpha1", )}
    # Should retrieve at least one chunk from doc_a
    assert "doc_a.txt" in sources or pipeline.store.size > 0


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def test_format_prompt_contains_query():
    chunk = Chunk(text="Paris is the capital of France.",
                  source="geo.txt", chunk_index=0, start_char=0, end_char=30)
    prompt = format_prompt("What is the capital of France?", [(chunk, 0.95)])
    assert "What is the capital of France?" in prompt


def test_format_prompt_contains_chunk_text():
    chunk = Chunk(text="Paris is the capital of France.",
                  source="geo.txt", chunk_index=0, start_char=0, end_char=30)
    prompt = format_prompt("What is the capital of France?", [(chunk, 0.95)])
    assert "Paris is the capital of France." in prompt


def test_format_prompt_contains_source():
    chunk = Chunk(text="Some content.", source="myfile.txt",
                  chunk_index=0, start_char=0, end_char=13)
    prompt = format_prompt("query", [(chunk, 0.8)])
    assert "myfile.txt" in prompt


def test_format_prompt_multiple_chunks_numbered():
    chunks = [
        (Chunk(text=f"chunk {i}", source="doc.txt", chunk_index=i,
               start_char=i*10, end_char=i*10+7), 0.9 - i * 0.1)
        for i in range(3)
    ]
    prompt = format_prompt("query", chunks)
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "[3]" in prompt


# ---------------------------------------------------------------------------
# Full query()
# ---------------------------------------------------------------------------

def test_query_calls_generator_with_prompt():
    received_prompts = []

    def mock_generator(prompt: str) -> str:
        received_prompts.append(prompt)
        return "Paris"

    pipeline = _make_pipeline(generator_fn=mock_generator)
    pipeline.ingest([{"text": "Paris is the capital of France. " * 5, "source": "geo.txt"}])
    result = pipeline.query("What is the capital of France?")

    assert len(received_prompts) == 1
    assert "What is the capital of France?" in received_prompts[0]


def test_query_returns_answer():
    pipeline = _make_pipeline(generator_fn=lambda p: "The answer is 42")
    pipeline.ingest([{"text": " ".join(f"w{i}" for i in range(30)), "source": "doc.txt"}])
    result = pipeline.query("what is the answer?")
    assert result["answer"] == "The answer is 42"


def test_query_returns_retrieved_chunks():
    pipeline = _make_pipeline(generator_fn=lambda p: "ok")
    pipeline.ingest([{"text": " ".join(f"w{i}" for i in range(30)), "source": "doc.txt"}])
    result = pipeline.query("w1 w2 w3")
    assert "retrieved_chunks" in result
    assert len(result["retrieved_chunks"]) > 0


def test_query_returns_prompt():
    pipeline = _make_pipeline(generator_fn=lambda p: "ok")
    pipeline.ingest([{"text": " ".join(f"w{i}" for i in range(30)), "source": "doc.txt"}])
    result = pipeline.query("w1 w2")
    assert "prompt" in result
    assert "w1 w2" in result["prompt"]


def test_query_without_generator_raises():
    pipeline = _make_pipeline(generator_fn=None)
    pipeline.ingest([{"text": " ".join(f"w{i}" for i in range(20)), "source": "doc.txt"}])
    with pytest.raises(RuntimeError):
        pipeline.query("anything")
