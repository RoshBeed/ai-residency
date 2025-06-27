FROM python:3.13.3-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Copy requirements from root directory
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project for access to transformer module and config
COPY . .

# Create model_data directory
RUN mkdir -p model_data

# Default command for classifier training
CMD ["python", "models/classifier/trainer.py"] 