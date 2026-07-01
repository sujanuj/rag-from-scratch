"""Sentence embeddings using a pretrained model from HuggingFace.

Uses sentence-transformers/all-MiniLM-L6-v2 -- a small (22M parameter),
fast model that produces 384-dimensional embeddings. It's the standard
baseline for semantic search: good enough to retrieve relevant chunks on
most corpora, fast enough to embed thousands of chunks in seconds on CPU.

Why not build embeddings from scratch? Unlike the chunker (pure string
ops) or the vector store (pure NumPy), the embedding model's value comes
entirely from pretraining on hundreds of millions of sentence pairs. A
from-scratch embedding model trained here would produce random-quality
vectors -- not a useful baseline. The honest choice is to use a
pretrained model and focus the "from scratch" story on the retrieval
and serving infrastructure around it.

The Embedder class wraps the model with:
  - batch processing (embed many chunks at once, not one at a time)
  - L2 normalization (so cosine similarity reduces to dot product,
    which is faster to compute in the vector store)
  - a measured embed_chunks() method that returns embeddings aligned
    with the input chunk list, so the vector store can store them
    together without a separate indexing step.
"""

from __future__ import annotations

from typing import List

import numpy as np


class Embedder:
    """Wraps a sentence-transformers model for chunk embedding.

    Args:
        model_name: HuggingFace model name. Defaults to
            all-MiniLM-L6-v2 (384-dim, 22M params, fast on CPU).
        batch_size: number of texts to embed in one forward pass.
            Larger batches are more efficient but use more memory.
            64 is a safe default for CPU inference.
        normalize: if True (default), L2-normalize all embeddings so
            cosine similarity == dot product. The vector store
            (retriever/vector_store.py) relies on this -- it uses dot
            product for speed and assumes unit vectors.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None  # lazy-loaded on first use

    def _load(self):
        """Load the model on first use rather than at construction time.
        This keeps import time fast and lets tests that don't need real
        embeddings avoid the download entirely.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors this model produces.
        384 for all-MiniLM-L6-v2. Used by the vector store to
        pre-allocate its matrix at the right width.
        """
        self._load()
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts and return an (N, dim) float32 array.

        Args:
            texts: list of strings to embed. Empty strings are allowed
                but will produce near-zero vectors after normalization --
                the caller should filter them out before ingestion if
                that's not desired.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32.
            Each row is a unit vector if normalize=True (default).
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings.astype(np.float32)

    def embed_chunks(self, chunks) -> np.ndarray:
        """Embed a list of Chunk objects (from ingestion/chunker.py).

        Convenience wrapper: extracts chunk.text, calls embed(), and
        returns the aligned (N, dim) array. The i-th row corresponds
        to chunks[i], so the vector store can zip(chunks, embeddings)
        without a separate alignment step.
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed(texts)
