"""
API server for Vision Transformer models.
Supports both regular ViT classification and seq2seq models.
"""

import os
import base64
import io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIG, MODEL_PATHS
from models.classifier.classifier import VisionTransformer
from models.seq2seq.seq2seq import Seq2SeqVisionTransformer

app = FastAPI(title="Vision Transformer API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_data: str
    model_type: str = "vit"  # "vit" or "seq2seq"

class PredictionResponse(BaseModel):
    pred: str
    confidence: float
    model_type: str

# Global model instances
vit_model = None
seq2seq_model = None
device = None

def load_models():
    """Load both ViT and seq2seq models"""
    global vit_model, seq2seq_model, device
    
    # Get device
    try:
        from utils.device_utils import get_torch_device_object
        device = get_torch_device_object("auto")  # Will auto-detect
    except ImportError:
        device = torch.device("cpu")
    
    print(f"🔧 Using device: {device}")
    
    try:
        # Load regular ViT model
        if os.path.exists(MODEL_PATHS["best_model"]):
            vit_model = VisionTransformer(**MODEL_CONFIG)
            vit_model.load_state_dict(torch.load(MODEL_PATHS["best_model"], map_location=device))
            vit_model.to(device)
            vit_model.eval()
            print(f"✅ Loaded ViT model from {MODEL_PATHS['best_model']}")
        else:
            print(f"⚠️  ViT model not found at {MODEL_PATHS['best_model']}")
        
        # Load seq2seq model
        if os.path.exists(MODEL_PATHS["best_seq2seq_model"]):
            seq2seq_model = Seq2SeqVisionTransformer(**MODEL_CONFIG)
            seq2seq_model.load_state_dict(torch.load(MODEL_PATHS["best_seq2seq_model"], map_location=device))
            seq2seq_model.to(device)
            seq2seq_model.eval()
            print(f"✅ Loaded Seq2Seq model from {MODEL_PATHS['best_seq2seq_model']}")
        else:
            print(f"⚠️  Seq2Seq model not found at {MODEL_PATHS['best_seq2seq_model']}")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")

def preprocess_image(image_data: str) -> torch.Tensor:
    """Preprocess base64 image data for model input"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to model input size
        image = image.resize((MODEL_CONFIG["img_size"], MODEL_CONFIG["img_size"]))
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch and channel dimensions and convert to tensor
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return image_tensor.to(device)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

def predict_vit(image_tensor: torch.Tensor) -> tuple[str, float]:
    """Predict using regular ViT model"""
    if vit_model is None:
        raise HTTPException(status_code=500, detail="ViT model not loaded")
    
    try:
        with torch.no_grad():
            # Get model prediction
            logits = vit_model(image_tensor)
            probabilities = F.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class = int(torch.argmax(logits, dim=-1).item())
            confidence = float(torch.max(probabilities, dim=-1)[0].item())
            
            return str(predicted_class), confidence
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ViT prediction failed: {e}")

def predict_seq2seq(image_tensor: torch.Tensor) -> tuple[str, float]:
    """Predict using seq2seq model"""
    if seq2seq_model is None:
        raise HTTPException(status_code=500, detail="Seq2Seq model not loaded")
    
    try:
        with torch.no_grad():
            # Get model prediction
            output_sequence = seq2seq_model.generate(image_tensor, max_length=5)
            
            # Convert sequence to string (remove START token and everything after EOS)
            digits = []
            for token in output_sequence[0]:  # Remove batch dimension
                token_int = int(token.item())
                if token_int == MODEL_CONFIG["start_token"]:
                    continue
                elif token_int == MODEL_CONFIG["eos_token"]:
                    break
                else:
                    digits.append(str(token_int))
            
            prediction = "".join(digits) if digits else "0"
            
            # For seq2seq, we'll use a fixed confidence (could be improved with beam search)
            confidence = 0.85
            
            return prediction, confidence
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seq2Seq prediction failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Vision Transformer API",
        "models_loaded": {
            "vit": vit_model is not None,
            "seq2seq": seq2seq_model is not None
        },
        "device": str(device) if device else "unknown"
    }

@app.get("/models")
async def get_models():
    """Get available models"""
    return {
        "available_models": [
            {"id": "vit", "name": "Vision Transformer", "type": "classification", "loaded": vit_model is not None},
            {"id": "seq2seq", "name": "Seq2Seq Vision Transformer", "type": "sequence", "loaded": seq2seq_model is not None}
        ],
        "device": str(device) if device else "unknown"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    """Predict digits from image"""
    try:
        # Preprocess image
        image_tensor = preprocess_image(request.image_data)
        
        # Choose model based on request
        if request.model_type == "seq2seq":
            if seq2seq_model is None:
                raise HTTPException(status_code=500, detail="Seq2Seq model not loaded")
            prediction, confidence = predict_seq2seq(image_tensor)
            model_type = "seq2seq"
        else:  # Default to ViT
            if vit_model is None:
                raise HTTPException(status_code=500, detail="ViT model not loaded")
            prediction, confidence = predict_vit(image_tensor)
            model_type = "vit"
        
        return PredictionResponse(
            pred=prediction,
            confidence=confidence,
            model_type=model_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict/vit", response_model=PredictionResponse)
async def predict_vit_endpoint(request: ImageRequest):
    """Predict using ViT model specifically"""
    request.model_type = "vit"
    return await predict(request)

@app.post("/predict/seq2seq", response_model=PredictionResponse)
async def predict_seq2seq_endpoint(request: ImageRequest):
    """Predict using seq2seq model specifically"""
    request.model_type = "seq2seq"
    return await predict(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 