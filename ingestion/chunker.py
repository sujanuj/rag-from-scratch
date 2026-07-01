"""Document ingestion and chunking pipeline.

Takes raw text (from files or strings), splits it into overlapping
fixed-size chunks, and attaches metadata (source, chunk index, character
offsets) to each chunk. No external library -- just plain string
operations.

Why overlapping chunks? A chunk boundary might fall mid-sentence,
cutting a fact in half. Overlapping by `overlap` tokens means every
sentence appears fully in at least one chunk, so retrieval doesn't miss
an answer because it was split across a boundary.

Chunk size and overlap are explicit parameters rather than hardcoded
constants because the right values depend on the corpus and the
embedding model's context window. The benchmark in
benchmark/measure_chunking.py measures how these choices affect chunk
count and coverage so the tradeoff is visible rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Chunk:
    """A single chunk of text with its metadata.

    text: the chunk's content, ready to be embedded.
    source: filename or identifier of the document this chunk came from.
    chunk_index: position of this chunk within its source document
        (0-based). Used to reconstruct reading order if needed.
    start_char: character offset of the first character of this chunk
        in the original document text. Together with end_char, lets
        callers locate the chunk's exact position in the source.
    end_char: character offset one past the last character of this chunk.
    """
    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return (f"Chunk(source={self.source!r}, index={self.chunk_index}, "
                f"chars={self.start_char}:{self.end_char}, text={preview!r}...)")


def split_into_words(text: str) -> List[tuple]:
    """Split text into (word, start_char, end_char) triples.

    Splitting on whitespace boundaries rather than fixed character counts
    so chunks never break in the middle of a word. Each triple carries
    the word's character offsets in the original text so chunk start_char
    and end_char can be computed exactly.
    """
    words = []
    for m in re.finditer(r'\S+', text):
        words.append((m.group(), m.start(), m.end()))
    return words


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 256,
    overlap: int = 32,
) -> List[Chunk]:
    """Split text into overlapping chunks of approximately chunk_size words.

    Args:
        text: raw document text to chunk.
        source: identifier attached to every chunk (filename, URL, etc).
        chunk_size: target number of words per chunk. The last chunk may
            be shorter if the document doesn't divide evenly.
        overlap: number of words from the end of chunk N that are repeated
            at the start of chunk N+1. Must be less than chunk_size.

    Returns:
        List of Chunk objects in document order.

    Raises:
        ValueError: if overlap >= chunk_size (overlap must be strictly
            smaller, or every chunk would be a subset of the previous one).
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
    if not text.strip():
        return []

    words = split_into_words(text)
    if not words:
        return []

    chunks = []
    chunk_index = 0
    pos = 0  # current word position in the words list

    while pos < len(words):
        end = min(pos + chunk_size, len(words))
        window = words[pos:end]

        chunk_text_str = text[window[0][1]:window[-1][2]]
        start_char = window[0][1]
        end_char = window[-1][2]

        chunks.append(Chunk(
            text=chunk_text_str,
            source=source,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
        ))

        chunk_index += 1

        # Advance by chunk_size - overlap words so the next chunk
        # starts overlap words before where this chunk ended.
        step = chunk_size - overlap
        pos += step

    return chunks


def chunk_file(
    path: str | Path,
    chunk_size: int = 256,
    overlap: int = 32,
    encoding: str = "utf-8",
) -> List[Chunk]:
    """Read a text file and return its chunks.

    The source field on each chunk is set to the file's name (not its
    full path) so it's readable in retrieval results without exposing
    filesystem layout.
    """
    path = Path(path)
    text = path.read_text(encoding=encoding)
    return chunk_text(text, source=path.name, chunk_size=chunk_size, overlap=overlap)


def chunk_documents(
    documents: List[dict],
    chunk_size: int = 256,
    overlap: int = 32,
) -> List[Chunk]:
    """Chunk a list of documents given as dicts with 'text' and 'source' keys.

    Convenience wrapper for ingesting multiple documents at once -- the
    retriever (retriever/vector_store.py) accepts a flat list of Chunk
    objects regardless of how many source documents they came from.
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(
            doc["text"],
            source=doc.get("source", "unknown"),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks
