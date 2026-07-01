# Dockerfile for rag-from-scratch
#
# Build:  docker build -t rag-from-scratch .
# Run:    docker run -p 8000:8000 rag-from-scratch
# With HF key: docker run -p 8000:8000 -e HF_API_KEY=your_key rag-from-scratch

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching:
# requirements rarely change, so this layer is reused on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn sentence-transformers

# Pre-download the embedding model so the container doesn't need
# network access at runtime. The model is ~90MB and baked into the
# image -- a deliberate tradeoff: larger image, faster cold start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY ingestion/ ingestion/
COPY embeddings/ embeddings/
COPY retriever/ retriever/
COPY pipeline/ pipeline/
COPY server/ server/

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
