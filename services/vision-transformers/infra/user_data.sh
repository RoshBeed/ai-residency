#!/bin/bash

# Exit on any error
set -e

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Variables from Terraform
HF_TOKEN="${hf_token}"
HF_REPO="${hf_repo}"
GIT_REPO="${git_repo}"
TRAINING_TYPE="${training_type}"
MAX_TRAINING_TIME="${max_training_time}"

log "Starting transformer training setup..."

# Update system
log "Updating system packages..."
apt-get update
apt-get upgrade -y

# Install required packages
log "Installing required packages..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    unzip \
    htop \
    nvtop \
    nvidia-cuda-toolkit \
    nvidia-driver-535

# Install Docker
log "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install NVIDIA Docker runtime
log "Installing NVIDIA Docker runtime..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
systemctl restart docker

# Verify GPU access
log "Verifying GPU access..."
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Create training directory
log "Creating training directory..."
mkdir -p /opt/transformer-training
cd /opt/transformer-training

# Clone the repository
log "Cloning repository: $GIT_REPO"
git clone $GIT_REPO .
chown -R ubuntu:ubuntu /opt/transformer-training

# Switch to ubuntu user for training
su - ubuntu << 'EOF'

# Set up Python environment
log "Setting up Python environment..."
cd /opt/transformer-training
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
log "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
log "Installing project dependencies..."
pip install -r requirements.txt
pip install -r transformer/requirements.txt
pip install -r streamlit/requirements.txt

# Install Hugging Face Hub
pip install huggingface_hub

# Set up Hugging Face token
log "Setting up Hugging Face token..."
mkdir -p ~/.cache/huggingface
echo "token: $HF_TOKEN" > ~/.cache/huggingface/token

# Create training script
cat > train_and_upload.sh << 'TRAIN_SCRIPT'
#!/bin/bash

set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to upload model to Hugging Face
upload_to_hf() {
    local model_path=$1
    local model_name=$2
    
    log "Uploading $model_name to Hugging Face..."
    
    # Create model card
    cat > README.md << 'MODEL_CARD'
---
language: en
tags:
- vision
- transformer
- image-classification
- sequence-to-sequence
license: mit
---

# Vision Transformer for Multi-Digit Recognition

This model is trained on multi-digit MNIST data for recognizing 3-digit numbers (0-999).

## Model Architecture
- Vision Transformer (ViT) with classification head
- Sequence-to-Sequence ViT for digit-by-digit generation
- 84x84 grayscale input images
- 16x16 patch size
- 256 embedding dimension
- 6 encoder layers, 8 attention heads

## Training
- Dataset: Multi-digit MNIST (0-999)
- Optimizer: AdamW with warmup cosine scheduler
- Regularization: Dropout, label smoothing, gradient clipping

## Usage
The model can be used for both direct classification and sequence-to-sequence generation of multi-digit numbers.
MODEL_CARD

    # Upload to Hugging Face
    huggingface-cli upload $HF_REPO $model_path --token $HF_TOKEN
    
    log "Successfully uploaded $model_name"
}

# Function to run training with timeout
run_training_with_timeout() {
    local training_script=$1
    local timeout_hours=$2
    
    log "Starting training with $timeout_hours hour timeout..."
    
    # Start training in background
    timeout ${timeout_hours}h bash $training_script &
    TRAINING_PID=$!
    
    # Wait for training to complete or timeout
    wait $TRAINING_PID
    TRAINING_EXIT_CODE=$?
    
    if [ $TRAINING_EXIT_CODE -eq 124 ]; then
        log "Training timed out after $timeout_hours hours"
        return 1
    elif [ $TRAINING_EXIT_CODE -eq 0 ]; then
        log "Training completed successfully"
        return 0
    else
        log "Training failed with exit code $TRAINING_EXIT_CODE"
        return 1
    fi
}

# Main training execution
log "Starting transformer training pipeline..."

# Create model_data directory if it doesn't exist
mkdir -p model_data

# Run training based on type
case $TRAINING_TYPE in
    "vit")
        log "Training Vision Transformer (classification)..."
        run_training_with_timeout "python vit_encoder/train_vit_encoder.py" $MAX_TRAINING_TIME
        if [ $? -eq 0 ]; then
            upload_to_hf "model_data/best_multi_digit_vit.pth" "vit-classification"
        fi
        ;;
    "seq2seq")
        log "Training Seq2Seq Vision Transformer..."
        run_training_with_timeout "python s2s_vit/train_seq2seq_vit.py" $MAX_TRAINING_TIME
        if [ $? -eq 0 ]; then
            upload_to_hf "model_data/best_multi_digit_vit_seq2seq.pth" "vit-seq2seq"
        fi
        ;;
    "both")
        log "Training both ViT and Seq2Seq models..."
        
        # Train ViT first
        log "Training Vision Transformer..."
        run_training_with_timeout "python vit_encoder/train_vit_encoder.py" $((MAX_TRAINING_TIME/2))
        VIT_SUCCESS=$?
        
        # Train Seq2Seq
        log "Training Seq2Seq Vision Transformer..."
        run_training_with_timeout "python s2s_vit/train_seq2seq_vit.py" $((MAX_TRAINING_TIME/2))
        SEQ2SEQ_SUCCESS=$?
        
        # Upload successful models
        if [ $VIT_SUCCESS -eq 0 ]; then
            upload_to_hf "model_data/best_multi_digit_vit.pth" "vit-classification"
        fi
        
        if [ $SEQ2SEQ_SUCCESS -eq 0 ]; then
            upload_to_hf "model_data/best_multi_digit_vit_seq2seq.pth" "vit-seq2seq"
        fi
        ;;
esac

log "Training pipeline completed"

# Create completion marker
echo "Training completed at $(date)" > /opt/transformer-training/training_completed.txt

# Terminate instance after successful completion
log "Training completed successfully. Terminating instance..."
aws ec2 terminate-instances --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id) --region $(curl -s http://169.254.169.254/latest/meta-data/placement/region)

TRAIN_SCRIPT

chmod +x train_and_upload.sh

# Start training
log "Starting training pipeline..."
./train_and_upload.sh

EOF

log "Setup completed successfully" 