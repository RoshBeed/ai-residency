# API Services

This directory contains the FastAPI service for serving Vision Transformer models.

## Structure

```
services/
├── __init__.py              # Package initialization
├── serve.py                 # FastAPI server implementation
├── requirements.txt         # Service dependencies
├── Dockerfile               # Container configuration
└── README.md                # This file
```

## API Server (`serve.py`)

A FastAPI server that serves both Vision Transformer classification and Seq2Seq models.

### Features

- **Dual Model Support**: Serves both ViT classifier and Seq2Seq models
- **RESTful API**: Clean REST endpoints for predictions
- **Image Processing**: Handles base64 encoded images
- **Model Loading**: Automatic model loading on startup
- **Health Checks**: Built-in health check endpoints
- **CORS Support**: Cross-origin resource sharing enabled

### Endpoints

- `GET /`: Health check and model status
- `GET /models`: List available models and their status
- `POST /predict`: General prediction endpoint (supports both models)
- `POST /predict/vit`: ViT classification endpoint
- `POST /predict/seq2seq`: Seq2Seq prediction endpoint

### Request Format

```json
{
  "image_data": "base64_encoded_image_string",
  "model_type": "vit"  // "vit" or "seq2seq"
}
```

### Response Format

```json
{
  "pred": "123",
  "confidence": 0.95,
  "model_type": "vit"
}
```

## Dependencies

The service requires:
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `torch`: PyTorch for model inference
- `pillow`: Image processing
- `numpy`: Numerical computations
- `pydantic`: Data validation

## Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn services.serve:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# Build and run with docker-compose
docker-compose up model-serve

# Or build manually
docker build -f services/Dockerfile -t transformer-api .
docker run -p 8000:8000 transformer-api
```

## Configuration

The service uses the global configuration from `config.py`:
- Model paths and configurations
- Device detection
- API settings

## Model Loading

Models are loaded automatically on startup:
1. Checks for model files in `model_data/`
2. Loads models to the detected device (CUDA/MPS/CPU)
3. Sets models to evaluation mode
4. Reports loading status via health check endpoints 