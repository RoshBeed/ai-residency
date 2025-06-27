FROM python:3.13.3-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY services/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY config.py .
COPY utils/ utils/
COPY models/ models/
COPY transformer/ transformer/
COPY datasets/ datasets/
COPY services/serve.py .
COPY model_data/ model_data/

# Create model_data directory if it doesn't exist
RUN mkdir -p model_data

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"] 