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
  description = "Maximum hourly USD to bid for the spot instance. Empty (the default) bids the on-demand price, which is what AWS recommends: you still pay the current spot price, but the request is not silently unfulfillable when the market sits above a hardcoded cap."
  type        = string
  default     = ""
}

variable "volume_size" {
  description = "Size of the root volume in GB"
  type        = number
  default     = 100
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to reach port 22. Defaults to nothing: SSH is opt-in, and opening it to 0.0.0.0/0 exposes the box to the internet's background scanning within minutes. Pass your own address as a /32, e.g. \"203.0.113.7/32\" (`curl -s ifconfig.me`). Leave empty and reach the instance with SSM Session Manager instead, which needs no inbound rule at all."
  type        = string
  default     = ""

  validation {
    condition     = var.allowed_ssh_cidr != "0.0.0.0/0"
    error_message = "Refusing to open SSH to the whole internet. Pass your own address as a /32, or leave this empty and use SSM Session Manager."
  }
}

variable "ssh_key_name" {
  description = "Name of an existing EC2 key pair to attach, for SSH access. Empty (the default) attaches none, which is correct when reaching the instance through SSM Session Manager."
  type        = string
  default     = ""
}

variable "ssh_key_path" {
  description = "Path to the SSH private key file for accessing the instance"
  type        = string
  default     = "~/.ssh/id_rsa"
}

# The three secrets below are written to SSM Parameter Store as SecureString, and the
# instance reads them at boot through its IAM role. They are deliberately NOT passed
# through user_data: user_data is readable by any process on the box via the instance
# metadata service, and by anyone holding ec2:DescribeInstanceAttribute, so a secret
# placed there is a secret handed to everything running on that host.

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
  description = "Overrides the image's own default training entrypoint, e.g. a full module path for a service whose default stage isn't the one you want. Empty string (the default) runs the image's own default CMD."
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

variable "parameter_prefix" {
  description = "SSM Parameter Store path the three training secrets are written to and read back from, without leading or trailing slashes."
  type        = string
  default     = "ai-residency/training"
}

variable "ubuntu_ami_name" {
  description = "AMI name filter, pinned to one published build rather than tracking the newest. Bump deliberately; `aws ec2 describe-images --owners 099720109477 --filters 'Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*' --query 'sort_by(Images,&CreationDate)[-1].Name'` prints the current tip."
  type        = string
  default     = "ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-20240927"
}
