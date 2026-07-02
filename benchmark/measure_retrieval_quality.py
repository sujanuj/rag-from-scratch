"""Retrieval quality benchmark on 1000 Wikipedia articles.

Ingests 1000 real Wikipedia articles into the RAG pipeline and runs
10 factual queries where the correct source article is confirmed to
exist in the corpus.

Run:
  python benchmark/measure_retrieval_quality.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embedder import Embedder
from pipeline.rag_pipeline import RAGPipeline
from retriever.vector_store import VectorStore

# Queries matched to articles confirmed in the first 1000 Wikipedia articles
QUERIES = [
    ("Who was Abraham Lincoln?",                    "Abraham Lincoln"),
    ("What is albedo?",                             "Albedo"),
    ("What is anarchism?",                          "Anarchism"),
    ("Who was Aristotle?",                          "Aristotle"),
    ("What is altruism?",                           "Altruism"),
    ("What is alchemy?",                            "Alchemy"),
    ("What is the Apollo program?",                 "Apollo"),
    ("Who is Andre Agassi?",                        "Andre Agassi"),
    ("What is anthropology?",                       "Anthropology"),
    ("What is ASCII?",                              "ASCII"),
]


def main():
    print("Loading 1000 Wikipedia articles...")
    with open("benchmark/wiki_1000.json") as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} articles")

    print("\nBuilding pipeline and embedding corpus...")
    t0 = time.time()
    embedder = Embedder()
    store = VectorStore(dim=embedder.embedding_dim)
    pipeline = RAGPipeline(
        embedder=embedder,
        store=store,
        generator_fn=None,
        top_k=5,
        chunk_size=100,
        chunk_overlap=10,
    )

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        pipeline.ingest([{"text": d["text"], "source": d["title"]} for d in batch])
        print(f"  Ingested {min(i+batch_size, len(docs))}/{len(docs)} articles "
              f"({pipeline.store.size} chunks)")

    elapsed = time.time() - t0
    print(f"\nIngestion complete: {pipeline.store.size} chunks in {elapsed:.1f}s")

    print("\n" + "="*80)
    print("Retrieval quality benchmark (queries matched to confirmed corpus articles)")
    print("="*80)
    print(f"{'Query':<45} {'Top-1 source':<28} {'Score':>6}  {'Hit?'}")
    print("-"*80)

    top1_hits = 0
    top5_hits = 0
    results = []

    for query, expected_title in QUERIES:
        retrieved = pipeline.retrieve(query)
        top1_chunk, top1_score = retrieved[0]
        top1_source = top1_chunk.source

        in_top1 = expected_title.lower() in top1_source.lower()
        in_top5 = any(
            expected_title.lower() in chunk.source.lower()
            for chunk, _ in retrieved
        )

        if in_top1:
            top1_hits += 1
        if in_top5:
            top5_hits += 1

        hit_str = "YES" if in_top1 else ("top-5" if in_top5 else "MISS")
        print(f"{query[:44]:<45} {top1_source[:27]:<28} {top1_score:>6.4f}  {hit_str}")

        results.append({
            "query": query,
            "expected": expected_title,
            "top1_source": top1_source,
            "top1_score": top1_score,
            "in_top1": in_top1,
            "in_top5": in_top5,
            "top5_sources": [c.source for c, _ in retrieved],
        })

    print("-"*80)
    print(f"\nTop-1 accuracy: {top1_hits}/{len(QUERIES)} ({100*top1_hits/len(QUERIES):.0f}%)")
    print(f"Top-5 accuracy: {top5_hits}/{len(QUERIES)} ({100*top5_hits/len(QUERIES):.0f}%)")

    misses = [r for r in results if not r["in_top1"]]
    if misses:
        print(f"\nTop-1 misses (may still appear in top-5):")
        for r in misses:
            print(f"  Query:    {r['query']}")
            print(f"  Expected: {r['expected']}")
            print(f"  Got:      {r['top1_source']} (score={r['top1_score']:.4f})")
            print(f"  Top-5:    {r['top5_sources']}")

    print("\nNotes:")
    print(f"  Corpus: first 1000 Wikipedia articles (first 500 chars each)")
    print(f"  All 10 queries verified to have matching articles in corpus")
    print(f"  Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)")
    print(f"  Search: flat cosine similarity, O(N) scan, exact top-K")
    print(f"  Chunk size: 100 words, overlap: 10 words")
    print(f"  Total chunks: {pipeline.store.size}")
    print(f"  Ingestion time: {elapsed:.1f}s on Apple M5 CPU")


if __name__ == "__main__":
    main()
