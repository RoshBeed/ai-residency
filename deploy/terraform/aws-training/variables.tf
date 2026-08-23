variable "aws_region" {
  description = "AWS region to deploy the training instance"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type for training (should be GPU-enabled)"
  type        = string
  default     = "g4dn.xlarge" # 1x NVIDIA T4 GPU, 4 vCPUs, 16 GB RAM
}

variable "spot_price" {
  description = "Maximum spot price to pay for the instance (in USD per hour)"
  type        = string
  default     = "0.50" # $0.50 per hour max
}

variable "volume_size" {
  description = "Size of the root volume in GB"
  type        = number
  default     = 100
}

variable "ssh_key_path" {
  description = "Path to the SSH private key file for accessing the instance"
  type        = string
  default     = "~/.ssh/id_rsa"
}

variable "hf_token" {
  description = "Hugging Face API token. Training pushes its checkpoint here itself (see ai-residency-common's push_checkpoint) -- this module does not re-upload it separately."
  type        = string
  sensitive   = true
}

variable "wandb_api_key" {
  description = "W&B API key. Optional: without it, training still runs and logs offline (see ai-residency-common's init_tracking)."
  type        = string
  sensitive   = true
  default     = ""
}

variable "train_image" {
  description = "The pushed training image to pull and run, e.g. \"roshb/ai-residency-vit-classifier-train:latest\". Built from services/<name>/Dockerfile.train -- see `make build SERVICE=<name>` at the repository root."
  type        = string
}

variable "train_command" {
  description = "Overrides the image's own default training entrypoint, e.g. \"seq2seq\" for Week 3's second trainer, or a full module path for a week whose default stage isn't the one you want. Empty string (the default) runs the image's own default CMD."
  type        = string
  default     = ""
}

variable "docker_hub_username" {
  description = "Docker Hub username, only needed if train_image is a private repository."
  type        = string
  default     = ""
}

variable "docker_hub_token" {
  description = "Docker Hub access token, only needed if train_image is a private repository."
  type        = string
  sensitive   = true
  default     = ""
}

variable "max_training_time" {
  description = "Maximum training time in hours before auto-termination"
  type        = number
  default     = 24
}

variable "environment" {
  description = "Environment name for tagging"
  type        = string
  default     = "training"
}
