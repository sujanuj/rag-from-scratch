[![CI/CD](https://github.com/sujanuj/rag-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/sujanuj/rag-from-scratch/actions/workflows/ci.yml)

# rag-from-scratch

A retrieval-augmented generation (RAG) pipeline built from scratch in Python. No LangChain, no LlamaIndex, no Pinecone — every component (chunking, embedding, vector search, generation, serving, and deployment) is implemented directly so the mechanics are fully visible and measurable.

---

## What it does

Given a corpus of documents and a natural-language question, the pipeline:

1. **Ingests** documents by splitting them into overlapping fixed-size chunks (Phase 1).
2. **Embeds** each chunk into a dense vector using a pretrained sentence embedding model (Phase 2).
3. **Retrieves** the top-K most similar chunks using cosine similarity over a flat vector store (Phase 3).
4. **Generates** an answer by injecting retrieved chunks into a prompt and calling an LLM (Phase 4).
5. **Serves** the pipeline over HTTP via FastAPI `/ingest` and `/query` endpoints (Phase 5).
6. **Deploys** as a Docker container with an AWS EC2 deployment script (Phase 6).

---

## Benchmark results

**Retrieval quality on 1000 Wikipedia articles (all-MiniLM-L6-v2, Apple M5 CPU):**

```
Query                                         Top-1 source              Score   Hit?
Who was Abraham Lincoln?                      Abraham Lincoln           0.7062  YES
What is albedo?                               Albedo                    0.8154  YES
What is anarchism?                            Anarchism                 0.8509  YES
Who was Aristotle?                            Aristotle                 0.7344  YES
What is altruism?                             Altruism                  0.7850  YES
What is alchemy?                              Alchemy                   0.7284  YES
What is the Apollo program?                   Apollo program            0.6331  YES
Who is Andre Agassi?                          Andre Agassi              0.6442  YES
What is anthropology?                         Anthropology              0.7733  YES
What is ASCII?                                ASCII                     0.8435  YES

Top-1 accuracy:  10/10 (100%)
Top-5 accuracy:  10/10 (100%)
Corpus:          1000 Wikipedia articles, 1045 chunks
Ingestion time:  7.1s on Apple M5 CPU
```

**Honest scope note:** an earlier run with queries whose target articles were not in the 1000-article sample (e.g. "capital of France", "who invented the telephone") showed 30% accuracy — retrieval cannot return what was never ingested. The benchmark above uses queries verified against the actual corpus.

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
- [x] 2 slow integration tests verify real semantic similarity: similar
      sentences score higher than dissimilar ones.

**Phase 3: Flat vector store with cosine similarity search — done**

- [x] `retriever/vector_store.py` — pre-allocated (N, dim) NumPy matrix.
      Retrieves top-K chunks via dot product using `argpartition` for O(N)
      candidate selection before the final O(k log k) sort.
- [x] `save()` / `load()` persistence via `.npz`: embed once at ingest time,
      reload at startup, search at query time.
- [x] 14 tests: add/size tracking, exact-match retrieval, sorted results,
      save/load round-trip, chunk metadata preservation.

**Phase 4: RAG pipeline — done**

- [x] `pipeline/rag_pipeline.py` — wires chunker + embedder + vector store.
      `ingest()` chunks and embeds; `retrieve()` searches; `query()` formats
      a grounded prompt and calls the generator.
- [x] `make_hf_generator()` calls the HuggingFace Inference API (free tier)
      so generation works without a local GPU or checkpoint download.
- [x] 15 tests using mock embedder and mock generator.

**Phase 5: FastAPI server — done**

- [x] `server/app.py` — `POST /ingest`, `POST /query`, `GET /health`.
      Interactive Swagger UI available at `/docs`.
- [x] Input validation: empty documents, empty question, query before ingest
      all return 400 with clear error messages.
- [x] 10 tests using FastAPI TestClient with mock pipeline.

**Phase 6: Docker + deployment + retrieval benchmark — done**

- [x] `Dockerfile` — containerizes the FastAPI server with the embedding
      model pre-baked into the image. Verified locally:
      `docker build -t rag-from-scratch . && docker run -p 8000:8000 rag-from-scratch`
      returns `{"status":"ok"}` from `/health`.
- [x] `scripts/deploy.sh` — AWS EC2 deployment script (t2.micro, free tier).
      Creates security group, launches instance, installs Docker, and prints
      the commands to pull and run the container.
- [x] `benchmark/measure_retrieval_quality.py` — 10/10 top-1 retrieval
      accuracy on 1000 Wikipedia articles with real measured scores.

---

## Running tests

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers uvicorn
python -m pytest tests/ -v -m "not slow"   # 61 tests, no download needed
python -m pytest tests/ -v                  # 63 tests including real model
```

---

## Running the server locally

```bash
uvicorn server.app:app --reload --port 8000
```

Interactive API docs at `http://localhost:8000/docs`.

Set `HF_API_KEY` environment variable to enable real LLM generation.

---

## Running with Docker

```bash
docker build -t rag-from-scratch .
docker run -p 8000:8000 rag-from-scratch
# With HF API key:
docker run -p 8000:8000 -e HF_API_KEY=your_key rag-from-scratch
```

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
├── benchmark/
│   ├── measure_retrieval_quality.py  <- 10/10 accuracy on 1000 Wikipedia articles
│   └── wiki_1000.json               <- 1000 Wikipedia articles for benchmarking
├── scripts/
│   └── deploy.sh               <- AWS EC2 deployment script (Phase 6)
├── Dockerfile                  <- containerized server (Phase 6)
├── tests/                      <- one test file per module (61 fast tests)
└── requirements.txt
```

---

## Author

**Sujan Uppalli Jayadevappa**
MS Software Engineering — Arizona State University
GitHub: [sujanuj](https://github.com/sujanuj)
