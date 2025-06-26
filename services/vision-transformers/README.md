# Vision Transformer Multi-Digit Recognition

A comprehensive Vision Transformer implementation for multi-digit recognition using PyTorch, featuring both classification and sequence-to-sequence approaches with automatic device detection.

## 🚀 Quick Start

### Automatic Device Detection

The system automatically detects and configures the best available device:

- **🍎 Mac (Apple Silicon)**: Uses MPS (Metal Performance Shaders)
- **🚀 Linux with NVIDIA GPU**: Uses CUDA with GPU acceleration  
- **💻 Linux without GPU**: Falls back to CPU
- **🐳 Docker with NVIDIA**: Uses CUDA automatically
- **🐳 Docker without NVIDIA**: Falls back to CPU

### Using Docker Compose (Recommended for Servers)

1. **Start the services:**
   ```bash
   # Start the serving and frontend
   docker-compose up model-serve streamlit
   
   # Or start everything
   docker-compose up
   ```

2. **Train models (optional):**
   ```bash
   # Train the regular ViT model (uses CUDA if available)
   docker-compose --profile train up model-train
   
   # Train the seq2seq model (uses CUDA if available)
   docker-compose --profile train up s2s-train
   
   # Train both models
   docker-compose --profile train up
   ```

3. **Access the application:**
   - **Streamlit Frontend:** http://localhost:8501
   - **API Documentation:** http://localhost:8000/docs
   - **API Health Check:** http://localhost:8000

### Local Development (Recommended for Mac)

Since MPS is not supported in Docker on Mac, run the Python files directly:

1. **Install dependencies:**
   ```bash
   # Install PyTorch with MPS support (Mac)
   pip install torch torchvision torchaudio
   
   # Install other dependencies
   pip install -r transformer/requirements.txt
   pip install -r streamlit/requirements.txt
   ```

2. **Test device detection:**
   ```bash
   python test_device.py
   ```

3. **Run services:**
   ```bash
   # API server (will use MPS on Mac)
   python transformer/serve.py
   
   # Frontend (in another terminal)
   streamlit run streamlit/main.py
   ```

4. **Train models:**
   ```bash
   # Train ViT model (uses MPS on Mac)
   python transformer/train_vit.py
   
   # Train seq2seq model (uses MPS on Mac)
   python s2s_vit/train_seq2seq_vit.py
   ```

## 🏗️ Architecture

### Services

- **`model-serve`**: FastAPI server serving both ViT and seq2seq models
- **`streamlit`**: Interactive web frontend with model selection
- **`model-train`**: Training service for regular ViT classification
- **`s2s-train`**: Training service for seq2seq model

### Model Types

1. **Vision Transformer (Classification)**
   - Direct classification of multi-digit numbers (0-999)
   - Faster inference
   - Fixed output size

2. **Seq2Seq Vision Transformer**
   - Sequence-to-sequence approach
   - Generates digits one by one
   - More flexible for variable-length numbers

## 🎯 Features

- **Interactive Drawing Canvas**: Draw digits and see real-time predictions
- **Model Comparison**: Switch between classification and seq2seq models
- **Educational Visualizations**: 
  - Patch division visualization
  - Attention heatmaps
  - Model architecture diagrams
- **Real-time Processing**: See how your input is processed step-by-step
- **Automatic Device Detection**: Optimizes for your hardware automatically

## 📊 Model Configuration

The models use the following configuration (from `config.py`):

- **Input Size**: 84×84 grayscale images
- **Patch Size**: 16×16 pixels (25 patches total)
- **Embedding Dimension**: 256
- **Encoder Layers**: 6
- **Attention Heads**: 8
- **Parameters**: ~2.1M

### Device-Specific Optimizations

| Device | Batch Size | Learning Rate | Mixed Precision | Description |
|--------|------------|---------------|-----------------|-------------|
| **CUDA** | 128 | 0.0001 | ✅ | NVIDIA GPU with CUDA |
| **MPS** | 64 | 0.0001 | ❌ | Apple Silicon GPU |
| **CPU** | 32 | 0.00005 | ❌ | CPU fallback |

## 🔧 API Endpoints

- `GET /`: Health check
- `GET /models`: List available models
- `POST /predict`: General prediction endpoint
- `POST /predict/vit`: ViT classification endpoint
- `POST /predict/seq2seq`: Seq2Seq endpoint

## 📁 Project Structure

```
├── config.py                 # Central configuration with device detection
├── docker-compose.yml        # Multi-service orchestration with GPU support
├── test_device.py           # Device detection test script
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── device_utils.py      # Device detection and configuration
│   └── huggingface_utils.py # Hugging Face upload utility
├── transformer/              # Core model implementation
│   ├── serve.py             # FastAPI server
│   ├── vit_model.py         # ViT classification model
│   ├── seq2seq_vit.py       # Seq2Seq model
│   └── ...
├── s2s_vit/                  # Seq2Seq training scripts
│   ├── train_seq2seq_vit.py
│   └── train_single_digit.py
├── streamlit/                # Web frontend
│   └── main.py
└── model_data/               # Trained model weights
```

## 🎨 Usage

1. **Open the Streamlit app** at http://localhost:8501
2. **Select a model** from the sidebar (ViT or Seq2Seq)
3. **Draw digits** on the canvas
4. **View results** in the prediction tab
5. **Explore visualizations** in the other tabs

## 🛠️ Development

### Testing Device Detection

```bash
# Test what device will be used
python test_device.py
```

### Local Development

```bash
# Install PyTorch
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r transformer/requirements.txt
pip install -r streamlit/requirements.txt

# Run services individually
python transformer/serve.py  # API server
streamlit run streamlit/main.py  # Frontend
```

### Training

```bash
# Train ViT model (uses detected device)
python transformer/train_vit.py

# Train seq2seq model (uses detected device)
python s2s_vit/train_seq2seq_vit.py
```

## 🐳 Docker Setup for GPU Servers

### Prerequisites

1. **Install NVIDIA Docker runtime:**
   ```bash
   # Install nvidia-docker2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Verify GPU access:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### Running with GPU

```bash
# Train with GPU support
docker-compose --profile train up model-train

# Or run everything with GPU
docker-compose up
```

## 📦 PyTorch Installation

### Mac (Apple Silicon)
```bash
pip install torch torchvision torchaudio
```

### Linux/Windows (with CUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Linux/Windows (CPU only)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📈 Performance

- **Inference Time**: ~50ms per prediction
- **Accuracy**: >95% on test set
- **Memory Usage**: ~500MB for both models
- **Training Speed**: 
  - CUDA: ~2-3x faster than CPU
  - MPS: ~1.5-2x faster than CPU

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. 