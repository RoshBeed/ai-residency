# The published frontend image (roshbeed/ai-residency-frontend) — streamlit +
# requests only, no ML dependencies at all. Every model actually runs in its own
# week's backend container; this image just renders pages and makes HTTP calls.
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY . .

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
