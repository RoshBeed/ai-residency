FROM python:3.13.3-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project for access to models and config
COPY . .

# Create model_data directory
RUN mkdir -p model_data

EXPOSE 8000

CMD ["uvicorn", "services.serve:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"] 