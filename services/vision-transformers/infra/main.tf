terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_subnet" "default" {
  default_for_az = true
  vpc_id         = data.aws_vpc.default.id
}

data "aws_vpc" "default" {
  default = true
}

data "aws_security_group" "default" {
  name   = "default"
  vpc_id = data.aws_vpc.default.id
}

# IAM role for the instance
resource "aws_iam_role" "training_role" {
  name = "transformer-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# IAM instance profile
resource "aws_iam_instance_profile" "training_profile" {
  name = "transformer-training-profile"
  role = aws_iam_role.training_role.name
}

# Security group for the training instance
resource "aws_security_group" "training_sg" {
  name        = "transformer-training-sg"
  description = "Security group for transformer training instance"
  vpc_id      = data.aws_vpc.default.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "transformer-training-sg"
  }
}

# Spot instance request
resource "aws_spot_instance_request" "training_instance" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  spot_price            = var.spot_price
  wait_for_fulfillment  = true
  spot_type             = "one-time"
  instance_interruption_behavior = "terminate"

  subnet_id                   = data.aws_subnet.default.id
  vpc_security_group_ids      = [aws_security_group.training_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.training_profile.name
  associate_public_ip_address = true

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    hf_token = var.hf_token
    hf_repo  = var.hf_repo
    git_repo = var.git_repo
  }))

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
    encrypted   = true
  }

  tags = {
    Name = "transformer-training-instance"
    Project = "transformer-training"
  }

  # Wait for the instance to be ready
  provisioner "local-exec" {
    command = "echo 'Waiting for instance to be ready...'"
  }

  provisioner "remote-exec" {
    inline = ["echo 'Instance is ready'"]

    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file(var.ssh_key_path)
      host        = self.public_ip
    }
  }
}

# Output the instance details
output "instance_id" {
  description = "ID of the training instance"
  value       = aws_spot_instance_request.training_instance.id
}

output "public_ip" {
  description = "Public IP of the training instance"
  value       = aws_spot_instance_request.training_instance.public_ip
}

output "instance_type" {
  description = "Type of the training instance"
  value       = aws_spot_instance_request.training_instance.instance_type
}

output "spot_price" {
  description = "Spot price paid for the instance"
  value       = aws_spot_instance_request.training_instance.spot_price
} 