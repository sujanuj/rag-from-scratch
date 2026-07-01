"""RAG (Retrieval-Augmented Generation) pipeline.

Wires together the three components built in Phases 1-3:
  1. Chunker (ingestion/chunker.py) -- splits documents into chunks
  2. Embedder (embeddings/embedder.py) -- embeds chunks and queries
  3. VectorStore (retriever/vector_store.py) -- stores and retrieves chunks

And adds a generation step: given a query and the retrieved chunks,
formats a prompt and calls an LLM to produce a natural-language answer.

Generation backend:
  The pipeline uses the HuggingFace Inference API (a free tier exists)
  so no local GPU or checkpoint download is needed. The model is
  configurable; the default is mistralai/Mistral-7B-Instruct-v0.2,
  a strong open instruction-tuned model available on HF Hub.

  For testing without a real API key, the pipeline accepts a
  `generator_fn` override -- any callable that takes a prompt string
  and returns a string. This lets tests verify the full pipeline
  (ingest -> retrieve -> format -> generate) without network access.

Why HuggingFace Inference API rather than llama-inference?
  The llama-inference project requires downloading a ~2GB checkpoint
  and running a local server. The HF API lets the pipeline run
  immediately with no local GPU, which makes it deployable on a
  free-tier cloud instance (Phase 6). The architecture is the same
  either way -- swap the generator_fn to point at a local server
  and nothing else changes.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from ingestion.chunker import Chunk, chunk_documents, chunk_text
from embeddings.embedder import Embedder
from retriever.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(query: str, chunks: List[Tuple[Chunk, float]]) -> str:
    """Format a RAG prompt from a query and retrieved chunks.

    The prompt instructs the model to answer using only the provided
    context, and to say "I don't know" if the context doesn't contain
    the answer. This is the standard RAG prompt pattern -- grounding
    the model's answer in retrieved evidence rather than its training
    data.

    Args:
        query: the user's question.
        chunks: list of (Chunk, score) tuples from VectorStore.search(),
            in descending similarity order.

    Returns:
        A formatted prompt string ready to send to an LLM.
    """
    context_parts = []
    for i, (chunk, score) in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] (source: {chunk.source}, similarity: {score:.3f})\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)

    return (
        f"Answer the question using only the context below. "
        f"If the context does not contain the answer, say \"I don't know\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# HuggingFace Inference API generator
# ---------------------------------------------------------------------------

def make_hf_generator(
    model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    api_key: Optional[str] = None,
    max_new_tokens: int = 256,
) -> Callable[[str], str]:
    """Return a generator function that calls the HuggingFace Inference API.

    Args:
        model: HuggingFace model ID to use for generation.
        api_key: HF API key. If None, reads from the HF_API_KEY
            environment variable.
        max_new_tokens: maximum tokens to generate.

    Returns:
        A callable (prompt: str) -> str that calls the HF API and
        returns the generated text.
    """
    key = api_key or os.environ.get("HF_API_KEY", "")

    def generate(prompt: str) -> str:
        import urllib.request
        import json

        url = f"https://api-inference.huggingface.co/models/{model}"
        payload = json.dumps({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
            }
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        if isinstance(result, list) and result:
            return result[0].get("generated_text", "").strip()
        return str(result)

    return generate


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end RAG pipeline: ingest documents, answer queries.

    Args:
        embedder: Embedder instance for embedding chunks and queries.
        store: VectorStore instance for storing and retrieving chunks.
        generator_fn: callable (prompt: str) -> str for generation.
            If None, the pipeline can ingest and retrieve but not
            generate -- useful for testing retrieval in isolation.
        top_k: number of chunks to retrieve per query.
        chunk_size: words per chunk for ingestion.
        chunk_overlap: overlap words between consecutive chunks.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        generator_fn: Optional[Callable[[str], str]] = None,
        top_k: int = 5,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
    ):
        self.embedder = embedder
        self.store = store
        self.generator_fn = generator_fn
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, documents: List[dict]):
        """Ingest a list of documents into the pipeline.

        Args:
            documents: list of dicts with 'text' and 'source' keys.
                e.g. [{"text": "...", "source": "doc.txt"}]

        Each document is chunked, embedded, and added to the vector
        store. Ingestion is idempotent in the sense that calling ingest()
        twice adds the chunks twice -- deduplication is not implemented
        (noted as a scope limit in the README).
        """
        chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        if not chunks:
            return
        embeddings = self.embedder.embed_chunks(chunks)
        self.store.add(chunks, embeddings)

    def ingest_text(self, text: str, source: str = "inline"):
        """Convenience wrapper: ingest a single raw text string."""
        self.ingest([{"text": text, "source": source}])

    def retrieve(self, query: str) -> List[Tuple[Chunk, float]]:
        """Retrieve the top-K chunks most relevant to query.

        Returns:
            List of (Chunk, score) tuples sorted by descending similarity.
        """
        query_embedding = self.embedder.embed([query])[0]
        return self.store.search(query_embedding, top_k=self.top_k)

    def query(self, question: str) -> dict:
        """Answer a question using retrieved context.

        Returns a dict with:
          - answer: the generated answer string
          - retrieved_chunks: list of (Chunk, score) used as context
          - prompt: the formatted prompt sent to the generator

        Raises:
            RuntimeError: if no generator_fn is configured.
            ValueError: if the store is empty (nothing ingested yet).
        """
        if self.generator_fn is None:
            raise RuntimeError(
                "No generator_fn configured. Pass a generator_fn to "
                "RAGPipeline() or use retrieve() for retrieval only."
            )

        retrieved = self.retrieve(question)
        prompt = format_prompt(question, retrieved)
        answer = self.generator_fn(prompt)

        return {
            "answer": answer,
            "retrieved_chunks": retrieved,
            "prompt": prompt,
        }
