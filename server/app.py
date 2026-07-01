"""FastAPI server exposing the RAG pipeline over HTTP.

Two endpoints:
  POST /ingest   -- add documents to the pipeline
  POST /query    -- ask a question, get an answer with sources
  GET  /health   -- liveness check

The pipeline is initialized at startup with a mock generator by default
(so the server works without an HF API key). Pass HF_API_KEY as an
environment variable to enable real generation.

Run locally:
  uvicorn server.app:app --reload --port 8000

Then test:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -d '{"documents": [{"text": "Paris is the capital of France.", "source": "geo.txt"}]}'
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the capital of France?"}'
"""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embeddings.embedder import Embedder
from pipeline.rag_pipeline import RAGPipeline, make_hf_generator
from retriever.vector_store import VectorStore

app = FastAPI(
    title="RAG from Scratch",
    description="Retrieval-Augmented Generation pipeline built from scratch.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Pipeline initialization (runs once at startup)
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

def _build_pipeline() -> RAGPipeline:
    embedder = Embedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,
        normalize=True,
    )
    store = VectorStore(dim=_EMBEDDING_DIM)

    hf_key = os.environ.get("HF_API_KEY", "")
    if hf_key:
        generator_fn = make_hf_generator(api_key=hf_key)
    else:
        # No API key: generator returns a message explaining what to do.
        # The pipeline is still fully functional for ingest and retrieve.
        def generator_fn(prompt: str) -> str:
            return (
                "Generation disabled: set HF_API_KEY environment variable "
                "to enable LLM answers. Retrieved context is shown in sources."
            )

    return RAGPipeline(
        embedder=embedder,
        store=store,
        generator_fn=generator_fn,
        top_k=5,
        chunk_size=256,
        chunk_overlap=32,
    )


# Module-level pipeline instance
pipeline: RAGPipeline = None


@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = _build_pipeline()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Document(BaseModel):
    text: str
    source: str = "unknown"


class IngestRequest(BaseModel):
    documents: List[Document]


class IngestResponse(BaseModel):
    ingested: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class ChunkResult(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[ChunkResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": pipeline.store.size if pipeline else 0}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents list is empty")

    docs = [{"text": doc.text, "source": doc.source} for doc in request.documents]
    before = pipeline.store.size
    pipeline.ingest(docs)
    after = pipeline.store.size

    return IngestResponse(
        ingested=len(request.documents),
        total_chunks=after,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if pipeline.store.size == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. POST to /ingest first."
        )

    if request.top_k is not None:
        original_top_k = pipeline.top_k
        pipeline.top_k = request.top_k

    try:
        result = pipeline.query(request.question)
    finally:
        if request.top_k is not None:
            pipeline.top_k = original_top_k

    sources = [
        ChunkResult(
            text=chunk.text,
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            score=round(score, 4),
        )
        for chunk, score in result["retrieved_chunks"]
    ]

    return QueryResponse(answer=result["answer"], sources=sources)
