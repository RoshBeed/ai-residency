# Transformer Training Infrastructure

This directory contains Terraform configurations to provision AWS spot GPU instances for training Vision Transformers and automatically upload the trained models to Hugging Face.

## 🚀 Features

- **Spot GPU Instances**: Cost-effective training using AWS spot instances
- **Automatic Setup**: Complete environment setup with CUDA, PyTorch, and dependencies
- **Training Pipeline**: Automated training of both ViT and Seq2Seq models
- **Auto-upload**: Models automatically uploaded to Hugging Face upon completion
- **Auto-termination**: Instance terminates itself after training to save costs
- **Monitoring**: Easy access to training logs and instance status

## 📋 Prerequisites

1. **AWS Account** with appropriate permissions
2. **Terraform** installed (version >= 1.0)
3. **AWS CLI** configured with credentials
4. **SSH Key Pair** for accessing the instance
5. **Hugging Face Token** for uploading models

## 🛠️ Setup Instructions

### 1. Configure AWS Credentials

```bash
aws configure
```

### 2. Create SSH Key Pair (if you don't have one)

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/transformer-training
```

### 3. Get Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with write permissions
3. Copy the token (starts with `hf_`)

### 4. Configure Variables

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your specific values:

```hcl
# AWS Configuration
aws_region = "us-east-1"
instance_type = "g4dn.xlarge"
spot_price = "0.50"
volume_size = 100

# SSH Configuration
ssh_key_path = "~/.ssh/transformer-training"

# Hugging Face Configuration
hf_token = "hf_your_actual_token_here"
hf_repo = "your-username/transformer-models"

# Git Repository
git_repo = "https://github.com/your-username/MLX-Week3-Transformers.git"

# Training Configuration
training_type = "both"  # "vit", "seq2seq", or "both"
max_training_time = 24
```

### 5. Initialize Terraform

```bash
terraform init
```

### 6. Review the Plan

```bash
terraform plan
```

### 7. Deploy the Infrastructure

```bash
terraform apply
```

## 🎯 Training Options

### Training Types

- **`vit`**: Train only the Vision Transformer classification model
- **`seq2seq`**: Train only the Seq2Seq Vision Transformer model  
- **`both`**: Train both models (default)

### Instance Types

Recommended GPU instance types:

| Instance Type | GPU | vCPUs | RAM | Approx. Cost/Hour |
|---------------|-----|-------|-----|-------------------|
| `g4dn.xlarge` | 1x T4 | 4 | 16GB | $0.50 |
| `g4dn.2xlarge` | 1x T4 | 8 | 32GB | $0.75 |
| `g5.xlarge` | 1x A10G | 4 | 16GB | $1.00 |
| `p3.2xlarge` | 1x V100 | 8 | 61GB | $3.00 |

## 📊 Monitoring

### View Training Progress

```bash
# Get the SSH command from Terraform outputs
terraform output ssh_command

# View training logs in real-time
terraform output training_logs_command
```

### Check Instance Status

```bash
# View all outputs
terraform output

# Check instance state
aws ec2 describe-instances --instance-ids $(terraform output -raw instance_id)
```

### Cost Estimation

```bash
# View estimated costs
terraform output estimated_cost_per_hour
terraform output max_total_cost
```

## 🔧 Customization

### Modify Training Parameters

Edit the user data script (`user_data.sh`) to customize:

- Training epochs
- Batch sizes
- Learning rates
- Model configurations

### Add Additional Dependencies

Add packages to the installation section in `user_data.sh`:

```bash
apt-get install -y \
    your-package-1 \
    your-package-2
```

### Custom Training Scripts

Modify the training execution section to run custom scripts:

```bash
# Add your custom training commands
python your_custom_training_script.py
```

## 🧹 Cleanup

### Manual Cleanup

```bash
# Destroy the infrastructure
terraform destroy

# Clean up local files
rm -rf .terraform*
```

### Automatic Cleanup

The instance automatically terminates itself after:
- Training completes successfully
- Maximum training time is reached
- Training fails

## 🔍 Troubleshooting

### Common Issues

1. **Spot Instance Not Available**
   - Try a different region
   - Increase spot price
   - Use a different instance type

2. **Training Fails**
   - Check logs: `terraform output training_logs_command`
   - SSH into instance: `terraform output ssh_command`
   - Verify GPU access: `nvidia-smi`

3. **Upload to Hugging Face Fails**
   - Verify HF token is correct
   - Check repository permissions
   - Ensure repository exists

### Debug Commands

```bash
# SSH into instance
ssh -i ~/.ssh/transformer-training ubuntu@$(terraform output -raw public_ip)

# Check GPU status
nvidia-smi

# Check training logs
tail -f /opt/transformer-training/training.log

# Check system resources
htop
nvtop
```

## 📈 Cost Optimization

### Spot Instance Tips

- Use spot instances for significant cost savings (60-90% off on-demand)
- Set reasonable spot prices (check current prices in AWS console)
- Use multiple regions to increase availability
- Monitor spot instance interruption rates

### Instance Selection

- Start with `g4dn.xlarge` for testing
- Use `g5.xlarge` for better performance
- Use `p3.2xlarge` for large-scale training

## 🔐 Security

### Best Practices

- Use IAM roles instead of access keys
- Restrict SSH access to your IP
- Use encrypted volumes
- Regularly rotate SSH keys
- Monitor instance logs

### Network Security

The security group allows:
- SSH access (port 22) from anywhere
- All outbound traffic

Consider restricting SSH access to your IP address for production use.

## 📚 Additional Resources

- [AWS Spot Instances](https://aws.amazon.com/ec2/spot/)
- [Terraform Documentation](https://www.terraform.io/docs)
- [Hugging Face Hub](https://huggingface.co/docs/hub)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)

## 🤝 Support

For issues with:
- **Infrastructure**: Check Terraform logs and AWS console
- **Training**: Check training logs and GPU status
- **Uploads**: Verify Hugging Face token and repository permissions 