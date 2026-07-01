# rag-from-scratch

A retrieval-augmented generation (RAG) pipeline built from scratch in
Python. No LangChain, no LlamaIndex, no Pinecone — every component
(chunking, embedding, vector search, generation, serving) is implemented
directly so the mechanics are fully visible and measurable.

---

## What it does

Given a corpus of documents and a natural-language question, the pipeline:

1. **Ingests** documents by splitting them into overlapping fixed-size chunks (Phase 1).
2. **Embeds** each chunk into a dense vector using a pretrained sentence embedding model (Phase 2).
3. **Retrieves** the top-K most similar chunks using cosine similarity over a flat vector store (Phase 3).
4. **Generates** an answer by injecting retrieved chunks into a prompt and calling an LLM (Phase 4).
5. **Serves** the pipeline over HTTP via FastAPI `/ingest` and `/query` endpoints (Phase 5).

---

## Phases

**Phase 1: Document ingestion and chunking — done**

- [x] `ingestion/chunker.py` — splits raw text into overlapping word-boundary
      chunks. `chunk_size` and `overlap` are explicit parameters. Character
      offsets (`start_char`, `end_char`) on every chunk locate it exactly in
      the source document — verified in tests by extracting
      `text[start_char:end_char]` and comparing to `chunk.text`.
- [x] 14 tests: chunk count, overlap correctness, character offsets, edge
      cases (empty text, overlap >= chunk_size).

**Phase 2: Sentence embeddings — done**

- [x] `embeddings/embedder.py` — wraps `sentence-transformers/all-MiniLM-L6-v2`
      (384-dim, 22M params) with lazy loading, batch processing, and L2
      normalization so cosine similarity reduces to dot product in the vector store.
- [x] 8 fast tests with mock model (no download needed).
- [x] 2 slow integration tests verify real semantic similarity.

**Phase 3: Flat vector store with cosine similarity search — done**

- [x] `retriever/vector_store.py` — pre-allocated (N, dim) NumPy matrix.
      Retrieves top-K chunks via dot product using `argpartition` for O(N)
      candidate selection before the final O(k log k) sort.
- [x] `save()` / `load()` persistence via `.npz`.
- [x] 14 tests: add/size tracking, exact-match retrieval, sorted results,
      save/load round-trip, chunk metadata preservation.

**Phase 4: RAG pipeline — done**

- [x] `pipeline/rag_pipeline.py` — wires chunker + embedder + vector store.
      `ingest()` chunks and embeds; `retrieve()` searches; `query()` formats
      a grounded prompt and calls the generator.
- [x] `make_hf_generator()` calls the HuggingFace Inference API (free tier).
- [x] 15 tests using mock embedder and mock generator.

**Phase 5: FastAPI server — done**

- [x] `server/app.py` — `POST /ingest`, `POST /query`, `GET /health`.
- [x] Input validation: empty documents, empty question, query before ingest
      all return 400 with clear error messages.
- [x] 10 tests using FastAPI TestClient with mock pipeline.

**Planned:**

- [ ] Phase 6: Docker + AWS/GCP deployment with live public endpoint

---

## Running tests

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v -m "not slow"   # 61 tests, no download needed
python -m pytest tests/ -v                  # 63 tests including real model
```

---

## Running the server locally

```bash
pip install uvicorn
uvicorn server.app:app --reload --port 8000
```

Ingest a document:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"text": "Paris is the capital of France.", "source": "geo.txt"}]}'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

Set `HF_API_KEY` environment variable to enable real LLM generation.

---

## Project layout

```
rag-from-scratch/
├── ingestion/
│   └── chunker.py              <- word-boundary chunking with overlap (Phase 1)
├── embeddings/
│   └── embedder.py             <- sentence-transformers wrapper (Phase 2)
├── retriever/
│   └── vector_store.py         <- flat cosine similarity search, save/load (Phase 3)
├── pipeline/
│   └── rag_pipeline.py         <- ingest + retrieve + generate (Phase 4)
├── server/
│   └── app.py                  <- FastAPI /ingest, /query, /health (Phase 5)
├── benchmark/                  <- retrieval benchmarks (planned)
├── tests/                      <- one test file per module (61 fast tests)
└── requirements.txt
```

---

## Author

**Sujan Uppalli Jayadevappa**
MS Software Engineering — Arizona State University
