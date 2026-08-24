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
# `most_recent = true` would silently change the AMI under you between two applies of
# the same commit, which is the opposite of what this module is for. The name filter is
# pinned to one published build; bump it deliberately.
data "aws_ami" "ubuntu" {
  owners = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = [var.ubuntu_ami_name]
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
  name = "ai-residency-training-role"

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

# The three secrets, held in SSM Parameter Store as SecureString rather than passed
# through user_data. user_data is readable by every process on the instance through the
# metadata service and by anyone with ec2:DescribeInstanceAttribute, so a token placed
# there is a token shared with everything running on the box -- and it is stored
# unencrypted in the Terraform state besides.
resource "aws_ssm_parameter" "hf_token" {
  name  = "/${var.parameter_prefix}/hf_token"
  type  = "SecureString"
  value = var.hf_token
}

resource "aws_ssm_parameter" "wandb_api_key" {
  name  = "/${var.parameter_prefix}/wandb_api_key"
  type  = "SecureString"
  value = var.wandb_api_key == "" ? "unset" : var.wandb_api_key
}

resource "aws_ssm_parameter" "docker_hub_token" {
  name  = "/${var.parameter_prefix}/docker_hub_token"
  type  = "SecureString"
  value = var.docker_hub_token == "" ? "unset" : var.docker_hub_token
}

# What the instance may actually do. The role previously had a trust policy and no
# permissions at all -- it granted nothing, which made the instance profile decoration.
# Now it grants exactly two things: read these three parameters, and decrypt them.
resource "aws_iam_role_policy" "read_training_secrets" {
  name = "ai-residency-read-training-secrets"
  role = aws_iam_role.training_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["ssm:GetParameter", "ssm:GetParameters"]
        Resource = [
          aws_ssm_parameter.hf_token.arn,
          aws_ssm_parameter.wandb_api_key.arn,
          aws_ssm_parameter.docker_hub_token.arn,
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = "ssm.${var.aws_region}.amazonaws.com"
          }
        }
      },
    ]
  })
}

# Lets you reach the instance with `aws ssm start-session` and no inbound rule at all,
# which is why allowed_ssh_cidr can stay empty.
resource "aws_iam_role_policy_attachment" "ssm_managed" {
  role       = aws_iam_role.training_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# IAM instance profile
resource "aws_iam_instance_profile" "training_profile" {
  name = "ai-residency-training-profile"
  role = aws_iam_role.training_role.name
}

# Security group for the training instance
resource "aws_security_group" "training_sg" {
  name        = "ai-residency-training-sg"
  description = "Security group for the AI Residency portfolio's training instance"
  vpc_id      = data.aws_vpc.default.id

  # SSH, only if a CIDR was supplied. With allowed_ssh_cidr empty (the default) this
  # block is not emitted at all and the instance has no inbound rules whatsoever --
  # reach it with `aws ssm start-session --target <instance-id>` instead, which goes
  # out through the SSM agent rather than in through a port.
  dynamic "ingress" {
    for_each = var.allowed_ssh_cidr == "" ? [] : [var.allowed_ssh_cidr]
    content {
      description = "SSH from the operator's own address"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  # Outbound stays open: training pulls its image from Docker Hub, its data and weights
  # from the Hugging Face Hub, and pushes results back. Pinning that to CIDRs would
  # break the first time any of those moved CDN.
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ai-residency-training-sg"
  }
}

# Spot instance request
resource "aws_spot_instance_request" "training_instance" {
  ami                            = data.aws_ami.ubuntu.id
  instance_type                  = var.instance_type
  spot_price                     = var.spot_price == "" ? null : var.spot_price
  wait_for_fulfillment           = true
  spot_type                      = "one-time"
  instance_interruption_behavior = "terminate"
  key_name                       = var.ssh_key_name == "" ? null : var.ssh_key_name

  subnet_id                   = data.aws_subnet.default.id
  vpc_security_group_ids      = [aws_security_group.training_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.training_profile.name
  associate_public_ip_address = true

  # Every ${...} reference inside user_data.sh must be one of these keys, or
  # templatefile() fails at plan time -- see user_data.sh's own note on this.
  #
  # No secret appears here. The script fetches the three tokens from SSM at boot using
  # the instance role, so what is stored in user_data (and in the Terraform state, and
  # in DescribeInstanceAttribute) is only the *names* of the parameters.
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    aws_region          = var.aws_region
    parameter_prefix    = var.parameter_prefix
    train_image         = var.train_image
    train_command       = var.train_command
    max_training_time   = var.max_training_time
    docker_hub_username = var.docker_hub_username
  }))

  # IMDSv2 required, not optional. v1 answers an unauthenticated GET from anything that
  # can make an HTTP request from this host -- including a compromised dependency in the
  # training image -- and hands back the instance role's credentials. hop_limit 1 keeps
  # a container on this host from reaching it at all.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
    encrypted   = true
  }

  depends_on = [
    aws_ssm_parameter.hf_token,
    aws_ssm_parameter.wandb_api_key,
    aws_ssm_parameter.docker_hub_token,
    aws_iam_role_policy.read_training_secrets,
  ]

  tags = {
    Name    = "ai-residency-training-instance"
    Project = "ai-residency"
  }

  # No remote-exec provisioner. It only echoed a string, but it forced every apply to
  # hold an SSH key and reach port 22 -- which is what made an open SSH rule feel
  # mandatory. `wait_for_fulfillment` above already blocks until the instance exists,
  # and the training run reports itself through cloud-init's log.
}

# Outputs live in outputs.tf, not here -- the original version of this module defined
# instance_id/public_ip/instance_type/spot_price in *both* files, which is a real
# Terraform error (duplicate output definition) that would have failed `terraform
# validate` outright. Fixed by keeping exactly one definition of each, in outputs.tf.
