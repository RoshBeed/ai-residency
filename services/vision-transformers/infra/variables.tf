variable "aws_region" {
  description = "AWS region to deploy the training instance"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type for training (should be GPU-enabled)"
  type        = string
  default     = "g4dn.xlarge"  # 1x NVIDIA T4 GPU, 4 vCPUs, 16 GB RAM
}

variable "spot_price" {
  description = "Maximum spot price to pay for the instance (in USD per hour)"
  type        = string
  default     = "0.50"  # $0.50 per hour max
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
  description = "Hugging Face API token for uploading models"
  type        = string
  sensitive   = true
}

variable "hf_repo" {
  description = "Hugging Face repository name (format: username/repo-name)"
  type        = string
  default     = "your-username/transformer-models"
}

variable "git_repo" {
  description = "Git repository URL containing the training code"
  type        = string
  default     = "https://github.com/your-username/MLX-Week3-Transformers.git"
}

variable "training_type" {
  description = "Type of training to run: 'vit', 'seq2seq', or 'both'"
  type        = string
  default     = "both"
  validation {
    condition     = contains(["vit", "seq2seq", "both"], var.training_type)
    error_message = "Training type must be 'vit', 'seq2seq', or 'both'."
  }
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