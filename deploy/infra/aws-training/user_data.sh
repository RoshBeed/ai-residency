#!/bin/bash

# Exit on any error
set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Variables from Terraform (templatefile() in main.tf) -- every brace-style variable
# reference below must be one of these seven, passed explicitly in that call, or
# terraform apply fails immediately at plan time. That was a real, pre-existing bug in
# the original version of this script: it referenced three brace-style bash variables
# that were never passed to templatefile() (one of them was even a local variable
# inside a bash function, which templatefile() cannot see at all) -- this module could
# never have actually been applied. Fixed here by keeping every non-Terraform bash
# variable reference brace-free ($VAR, no braces) throughout the rest of this file.
HF_TOKEN="${hf_token}"
WANDB_API_KEY="${wandb_api_key}"
TRAIN_IMAGE="${train_image}"
TRAIN_COMMAND="${train_command}"
MAX_TRAINING_TIME="${max_training_time}"
DOCKER_HUB_USERNAME="${docker_hub_username}"
DOCKER_HUB_TOKEN="${docker_hub_token}"

log "Starting AI Residency portfolio training setup..."

log "Updating system packages..."
apt-get update -y
apt-get upgrade -y

log "Installing base packages..."
apt-get install -y curl wget htop nvtop nvidia-driver-535 ca-certificates gnupg

log "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu
rm -f get-docker.sh

log "Installing the NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

log "Verifying GPU access..."
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

if [ -n "$DOCKER_HUB_TOKEN" ]; then
    log "Logging in to Docker Hub ($TRAIN_IMAGE is private)..."
    echo "$DOCKER_HUB_TOKEN" | docker login --username "$DOCKER_HUB_USERNAME" --password-stdin
fi

log "Pulling training image: $TRAIN_IMAGE"
docker pull "$TRAIN_IMAGE"

mkdir -p /opt/ai-residency-training/checkpoints
chown -R ubuntu:ubuntu /opt/ai-residency-training

# Runs "$@" under a timeout, in hours -- the one bash function in this script, kept
# free of brace-style $${...} variable references throughout (see the note above).
run_with_timeout() {
    local timeout_hours="$1"
    shift
    log "Starting training with $timeout_hours hour timeout..."
    timeout "$timeout_hours"h "$@" &
    local pid=$!
    wait "$pid"
    local exit_code=$?
    if [ "$exit_code" -eq 124 ]; then
        log "Training timed out after $timeout_hours hours"
        return 1
    elif [ "$exit_code" -eq 0 ]; then
        log "Training completed successfully"
        return 0
    else
        log "Training failed with exit code $exit_code"
        return 1
    fi
}

log "Running training container..."
# The image's own entrypoint (docker/train-entrypoint.sh where a repo has one, e.g.
# Week 3's classifier/seq2seq dispatch) already picks the right training stage by
# default -- TRAIN_COMMAND only needs to be set to override that default, e.g. for
# Week 1's predictor-only run instead of its image's default cbow stage. Training
# itself pushes its own checkpoint to the Hub and logs to W&B (see ai-residency-common's
# push_checkpoint / init_tracking, used by every repo's train(config)) -- this
# script does not re-implement that upload, unlike the version it replaced.
if [ -n "$TRAIN_COMMAND" ]; then
    run_with_timeout "$MAX_TRAINING_TIME" \
        docker run --rm --gpus all \
            -v /opt/ai-residency-training/checkpoints:/app/checkpoints \
            -e HF_TOKEN="$HF_TOKEN" \
            -e WANDB_API_KEY="$WANDB_API_KEY" \
            "$TRAIN_IMAGE" $TRAIN_COMMAND
else
    run_with_timeout "$MAX_TRAINING_TIME" \
        docker run --rm --gpus all \
            -v /opt/ai-residency-training/checkpoints:/app/checkpoints \
            -e HF_TOKEN="$HF_TOKEN" \
            -e WANDB_API_KEY="$WANDB_API_KEY" \
            "$TRAIN_IMAGE"
fi

log "Training pipeline completed."
echo "Training completed at $(date)" > /opt/ai-residency-training/training_completed.txt

log "Terminating instance to stop the spot-price meter..."
aws ec2 terminate-instances \
    --instance-ids "$(curl -s http://169.254.169.254/latest/meta-data/instance-id)" \
    --region "$(curl -s http://169.254.169.254/latest/meta-data/placement/region)"
