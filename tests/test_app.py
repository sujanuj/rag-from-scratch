"""Tests for the FastAPI server.

Uses FastAPI's TestClient (built on httpx) to send real HTTP requests
to the app without starting a live server. The pipeline is replaced with
a mock that uses a tiny in-memory setup -- no model download, no API key.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_test_pipeline():
    """Build a real RAGPipeline with a mock embedder and mock generator."""
    from embeddings.embedder import Embedder
    from pipeline.rag_pipeline import RAGPipeline
    from retriever.vector_store import VectorStore

    dim = 32
    embedder = Embedder.__new__(Embedder)
    embedder.model_name = "mock"
    embedder.batch_size = 64
    embedder.normalize = True

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim

    def mock_encode(texts, **kwargs):
        rng = np.random.default_rng(seed=42)
        vecs = rng.standard_normal((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock_model.encode.side_effect = mock_encode
    embedder._model = mock_model

    store = VectorStore(dim=dim)
    return RAGPipeline(
        embedder=embedder,
        store=store,
        generator_fn=lambda prompt: "Mock answer",
        top_k=3,
        chunk_size=10,
        chunk_overlap=2,
    )


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import server.app as app_module

    # Set pipeline before AND after startup so the startup event cannot
    # overwrite it with a real pipeline that needs sentence_transformers.
    test_pipeline = _make_test_pipeline()
    app_module.pipeline = test_pipeline
    with TestClient(app_module.app) as c:
        app_module.pipeline = test_pipeline  # override post-startup
        yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_returns_chunk_count(client):
    assert "chunks_indexed" in client.get("/health").json()


def test_ingest_single_document(client):
    resp = client.post("/ingest", json={
        "documents": [{"text": " ".join(f"word{i}" for i in range(30)), "source": "doc.txt"}]
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ingested"] == 1
    assert body["total_chunks"] > 0


def test_ingest_multiple_documents(client):
    resp = client.post("/ingest", json={
        "documents": [
            {"text": " ".join(f"alpha{i}" for i in range(20)), "source": "a.txt"},
            {"text": " ".join(f"beta{i}" for i in range(20)), "source": "b.txt"},
        ]
    })
    assert resp.status_code == 200
    assert resp.json()["ingested"] == 2


def test_ingest_empty_documents_returns_400(client):
    assert client.post("/ingest", json={"documents": []}).status_code == 400


def test_ingest_increases_chunk_count(client):
    before = client.get("/health").json()["chunks_indexed"]
    client.post("/ingest", json={
        "documents": [{"text": " ".join(f"w{i}" for i in range(30)), "source": "x.txt"}]
    })
    after = client.get("/health").json()["chunks_indexed"]
    assert after > before


def test_query_returns_answer(client):
    client.post("/ingest", json={
        "documents": [{"text": " ".join(f"w{i}" for i in range(40)), "source": "doc.txt"}]
    })
    resp = client.post("/query", json={"question": "what is w1?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Mock answer"


def test_query_returns_sources(client):
    client.post("/ingest", json={
        "documents": [{"text": " ".join(f"w{i}" for i in range(40)), "source": "doc.txt"}]
    })
    resp = client.post("/query", json={"question": "w1 w2"})
    assert resp.status_code == 200
    for s in resp.json()["sources"]:
        assert "text" in s and "source" in s and "score" in s


def test_query_empty_question_returns_400(client):
    assert client.post("/query", json={"question": "   "}).status_code == 400


def test_query_before_ingest_returns_400(client):
    from embeddings.embedder import Embedder
    from pipeline.rag_pipeline import RAGPipeline
    from retriever.vector_store import VectorStore
    import server.app as app_module
    from fastapi.testclient import TestClient

    store = VectorStore(dim=32)
    app_module.pipeline = RAGPipeline(
        embedder=_make_test_pipeline().embedder,
        store=store,
        generator_fn=lambda p: "x",
        top_k=3,
    )
    with TestClient(app_module.app) as c:
        assert c.post("/query", json={"question": "anything"}).status_code == 400
