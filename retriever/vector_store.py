"""A flat vector store with cosine similarity search.

Stores chunk embeddings in a single (N, dim) NumPy matrix and retrieves
the top-K most similar chunks to a query vector using dot product -- which
equals cosine similarity when both vectors are L2-normalized (the Embedder
normalizes by default).

Why flat search (O(N) scan) rather than an approximate index like HNSW?
For corpora up to ~100K chunks, flat search is fast enough on CPU and
produces EXACT top-K results -- no approximation error, no index-build
cost, and no external dependency. The benchmark in
benchmark/measure_retrieval.py measures where flat search breaks down
so the O(N) cost is visible rather than assumed. An HNSW index (e.g.
via hnswlib) would be the natural next step for larger corpora, and is
noted as such in the README.

Storage layout:
  _matrix: (N, dim) float32 NumPy array, one row per stored chunk.
  _chunks: list of Chunk objects, aligned with _matrix rows.

Both grow together on add() calls. The matrix is pre-allocated in
blocks to avoid per-call reallocation (the same motivation as the paged
KV-cache in llama-inference: concatenation is expensive at scale).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ingestion.chunker import Chunk

# Pre-allocation block size: how many rows to reserve when the matrix
# needs to grow. 1024 is a reasonable default -- large enough that
# reallocation is rare, small enough that memory waste is bounded.
_BLOCK_SIZE = 1024


class VectorStore:
    """A flat in-memory vector store.

    Args:
        dim: embedding dimensionality. Must match the Embedder's
            embedding_dim. Set at construction so the internal matrix
            can be pre-allocated at the right width.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._size = 0  # number of chunks actually stored
        # Pre-allocate a block of rows; grow as needed.
        self._matrix = np.zeros((_BLOCK_SIZE, dim), dtype=np.float32)
        self._chunks: List[Chunk] = []

    @property
    def size(self) -> int:
        """Number of chunks currently stored."""
        return self._size

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to the store.

        Args:
            chunks: list of Chunk objects (from ingestion/chunker.py).
            embeddings: (len(chunks), dim) float32 array, one row per
                chunk. Must be L2-normalized (the Embedder does this by
                default) for dot product == cosine similarity to hold.

        Raises:
            ValueError: if len(chunks) != len(embeddings), or if
                embeddings.shape[1] != self.dim.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have the same length, "
                f"got {len(chunks)} chunks and {len(embeddings)} embeddings"
            )
        if len(embeddings) == 0:
            return
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embedding dim mismatch: store expects {self.dim}, "
                f"got {embeddings.shape[1]}"
            )

        n = len(chunks)
        # Grow the matrix if needed, in blocks to avoid per-call realloc.
        while self._size + n > len(self._matrix):
            extra = max(_BLOCK_SIZE, n)
            self._matrix = np.concatenate(
                [self._matrix, np.zeros((extra, self.dim), dtype=np.float32)],
                axis=0,
            )

        self._matrix[self._size:self._size + n] = embeddings.astype(np.float32)
        self._chunks.extend(chunks)
        self._size += n

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Return the top-K chunks most similar to query_embedding.

        Args:
            query_embedding: (dim,) or (1, dim) float32 array, L2-normalized.
            top_k: number of results to return. Clamped to self.size if
                top_k > self.size.

        Returns:
            List of (Chunk, score) tuples sorted by descending similarity.
            score is the cosine similarity (dot product of unit vectors),
            in [-1, 1].

        Raises:
            ValueError: if the store is empty or query_embedding has the
                wrong dimensionality.
        """
        if self._size == 0:
            raise ValueError("Cannot search an empty vector store.")

        query = np.asarray(query_embedding, dtype=np.float32).flatten()
        if query.shape[0] != self.dim:
            raise ValueError(
                f"query dim mismatch: store has dim={self.dim}, "
                f"query has dim={query.shape[0]}"
            )

        top_k = min(top_k, self._size)

        # Dot product against all stored vectors (O(N * dim) scan).
        # This is exact cosine similarity since both query and stored
        # vectors are L2-normalized.
        scores = self._matrix[:self._size] @ query  # (N,)

        # argpartition gives the top_k indices in O(N) rather than O(N log N)
        # full sort -- only the final top_k sort is O(top_k log top_k).
        if top_k < self._size:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
        else:
            top_indices = np.arange(self._size)

        # Sort the top_k candidates by descending score.
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    def save(self, path: str):
        """Persist the store to disk as a .npz file.

        Saves the embedding matrix and chunk metadata (text, source,
        chunk_index, start_char, end_char) so the store can be reloaded
        without re-embedding the corpus. This is the key persistence
        operation for the server: embed once at ingest time, reload at
        startup, search at query time.
        """
        import json
        chunk_meta = json.dumps([
            {
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in self._chunks
        ])
        np.savez_compressed(
            path,
            matrix=self._matrix[:self._size],
            chunk_meta=np.array([chunk_meta]),
            dim=np.array([self.dim]),
        )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a store previously saved with save().

        The loaded store is immediately searchable -- no re-embedding needed.
        """
        import json
        data = np.load(path + ".npz", allow_pickle=False)
        dim = int(data["dim"][0])
        store = cls(dim=dim)
        matrix = data["matrix"]
        chunk_meta = json.loads(str(data["chunk_meta"][0]))
        chunks = [
            Chunk(
                text=m["text"],
                source=m["source"],
                chunk_index=m["chunk_index"],
                start_char=m["start_char"],
                end_char=m["end_char"],
            )
            for m in chunk_meta
        ]
        store.add(chunks, matrix)
        return store
