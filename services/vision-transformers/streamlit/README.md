# Streamlit Web Frontend

This directory contains the Streamlit web application for interactive Vision Transformer predictions.

## Structure

```
streamlit/
├── main.py                  # Streamlit application
├── requirements.txt         # Frontend dependencies
├── Dockerfile               # Container configuration
└── README.md                # This file
```

## Web Application (`main.py`)

An interactive web interface for testing Vision Transformer models with real-time drawing and predictions.

### Features

- **Interactive Drawing Canvas**: Draw digits with mouse/touch
- **Model Selection**: Switch between ViT classifier and Seq2Seq models
- **Real-time Predictions**: Get instant predictions as you draw
- **Educational Visualizations**: 
  - Patch division visualization
  - Attention heatmaps
  - Model architecture diagrams
- **Multiple Input Methods**: Drawing canvas, file upload, and URL input
- **Responsive Design**: Works on desktop and mobile devices

### Pages

1. **Home**: Welcome and model selection
2. **Predict**: Interactive drawing and predictions
3. **Visualize**: Educational visualizations and model insights
4. **About**: Project information and usage instructions

### Model Integration

The frontend connects to the API service (`model-serve`) for predictions:
- `MODEL_URL`: General prediction endpoint
- `VIT_URL`: ViT classification endpoint
- `SEQ2SEQ_URL`: Seq2Seq prediction endpoint
- `MODELS_URL`: Model status endpoint

## Dependencies

The frontend requires:
- `streamlit`: Web application framework
- `opencv-python`: Image processing
- `pillow`: Image handling
- `numpy`: Numerical computations
- `requests`: HTTP client for API calls
- `plotly`: Interactive visualizations

## Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py --server.port 8501 --server.address 0.0.0.0
```

### Docker

```bash
# Build and run with docker-compose
docker-compose up streamlit

# Or build manually
docker build -f streamlit/Dockerfile -t transformer-frontend .
docker run -p 8501:8501 transformer-frontend
```

## Environment Variables

The application uses these environment variables:
- `MODEL_URL`: API endpoint for predictions
- `VIT_URL`: ViT model endpoint
- `SEQ2SEQ_URL`: Seq2Seq model endpoint
- `MODELS_URL`: Model status endpoint

## Features

### Drawing Canvas

- **Real-time Drawing**: Draw digits with mouse or touch
- **Canvas Controls**: Clear, undo, and adjust brush size
- **Image Processing**: Automatic preprocessing for model input
- **Visual Feedback**: See the processed image sent to the model

### Model Comparison

- **Side-by-side Testing**: Compare predictions from both models
- **Confidence Scores**: View prediction confidence for each model
- **Model Status**: Check if models are loaded and available

### Educational Content

- **Patch Visualization**: See how images are divided into patches
- **Attention Maps**: Visualize model attention patterns
- **Architecture Diagrams**: Understand the model structure
- **Training Insights**: Learn about the training process

## Configuration

The frontend automatically detects the API service and adapts to:
- Available models
- Model loading status
- API response formats
- Error conditions 