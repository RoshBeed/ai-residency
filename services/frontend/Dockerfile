# syntax=docker/dockerfile:1
#
# The published frontend image (roshbeed/ai-residency-frontend): streamlit and
# requests, no ML dependencies at all. Every model runs in its own service's
# container; this one renders pages and makes HTTP calls.
#
# It does not build FROM ai-residency-base, deliberately -- that image carries a
# torch install this one has no use for.
#
# The build context is the repository root, not this directory.
#
#   docker build -f services/frontend/Dockerfile -t roshbeed/ai-residency-frontend:latest .

FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir uv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock ./
# --- workspace manifests (generated: make sync-dockerfiles) ---
COPY libs/common/pyproject.toml libs/common/
COPY libs/vision-transformer/pyproject.toml libs/vision-transformer/
COPY services/_template/pyproject.toml services/_template/
COPY services/content-scoring/pyproject.toml services/content-scoring/
COPY services/frontend/pyproject.toml services/frontend/
COPY services/multimodal-captioning/pyproject.toml services/multimodal-captioning/
COPY services/rlhf-ppo/pyproject.toml services/rlhf-ppo/
COPY services/speech-emotion-recognition/pyproject.toml services/speech-emotion-recognition/
COPY services/vector-search-retrieval/pyproject.toml services/vector-search-retrieval/
COPY services/vit-classifier/pyproject.toml services/vit-classifier/
COPY services/vit-seq2seq/pyproject.toml services/vit-seq2seq/
COPY services/whisper-fine-tuning/pyproject.toml services/whisper-fine-tuning/
COPY services/word-embeddings/pyproject.toml services/word-embeddings/
# --- end workspace manifests ---
COPY libs/common libs/common
RUN uv sync --frozen --no-install-project --package ai-residency-frontend

COPY services/frontend services/frontend

WORKDIR /app/services/frontend
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
