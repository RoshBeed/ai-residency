#!/bin/bash

# AI Residency Portfolio AWS Training Infrastructure Quick Start Script

set -e

echo "🚀 AI Residency Portfolio AWS Training Quick Start"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    echo "========================"
    
    local missing_deps=()
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        missing_deps+=("terraform")
    else
        print_status "✓ Terraform installed: $(terraform version | head -n1)"
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        missing_deps+=("aws-cli")
    else
        print_status "✓ AWS CLI installed: $(aws --version)"
    fi
    
    # Check if AWS credentials are configured
    if ! aws sts get-caller-identity &> /dev/null; then
        print_warning "AWS credentials not configured. Please run 'aws configure' first."
        missing_deps+=("aws-credentials")
    else
        print_status "✓ AWS credentials configured"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        echo ""
        echo "Please install missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            case $dep in
                "terraform")
                    echo "  - Terraform: https://www.terraform.io/downloads.html"
                    ;;
                "aws-cli")
                    echo "  - AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
                    ;;
                "aws-credentials")
                    echo "  - Run: aws configure"
                    ;;
            esac
        done
        exit 1
    fi
    
    print_status "All prerequisites satisfied!"
    echo ""
}

# Function to get user input with validation
get_user_input() {
    local prompt="$1"
    local default="$2"
    local validation_func="$3"
    
    while true; do
        if [ -n "$default" ]; then
            read -p "$prompt [$default]: " input
            input=${input:-$default}
        else
            read -p "$prompt: " input
        fi
        
        if [ -n "$validation_func" ]; then
            if $validation_func "$input"; then
                echo "$input"
                return 0
            else
                print_error "Invalid input. Please try again."
            fi
        else
            echo "$input"
            return 0
        fi
    done
}

# Validation functions
validate_region() {
    local region="$1"
    aws ec2 describe-regions --region-names "$region" &> /dev/null
}

validate_instance_type() {
    local instance_type="$1"
    [[ "$instance_type" =~ ^(g4dn\.xlarge|g4dn\.2xlarge|g5\.xlarge|p3\.2xlarge)$ ]]
}

validate_spot_price() {
    local price="$1"
    [[ "$price" =~ ^[0-9]+\.?[0-9]*$ ]] && (( $(echo "$price > 0" | bc -l) ))
}

validate_hf_token() {
    local token="$1"
    [[ "$token" =~ ^hf_[A-Za-z0-9]+$ ]]
}

validate_train_image() {
    local image="$1"
    # A Docker image reference: repo/name, optionally with a :tag.
    [[ "$image" =~ ^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(:[a-zA-Z0-9_.-]+)?$ ]]
}

# Function to create SSH key if needed
setup_ssh_key() {
    print_header "SSH Key Setup"
    echo "=============="
    
    local ssh_key_path="$1"
    
    if [ -f "$ssh_key_path" ]; then
        print_status "SSH key already exists: $ssh_key_path"
        return 0
    fi
    
    print_warning "SSH key not found: $ssh_key_path"
    read -p "Create new SSH key? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Creating SSH key..."
        ssh-keygen -t rsa -b 4096 -f "$ssh_key_path" -N ""
        print_status "SSH key created: $ssh_key_path"
        print_warning "Make sure to add the public key to your AWS account if needed."
    else
        print_error "SSH key is required. Please create one manually or provide an existing key path."
        exit 1
    fi
}

# Function to get Hugging Face token
get_hf_token() {
    print_header "Hugging Face Setup"
    echo "===================="

    echo "Training pushes its own checkpoint to the Hub as it runs (see ai-residency-common's"
    echo "push_checkpoint) -- this needs a write-scoped token."
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token with write permissions"
    echo "3. Copy the token (starts with 'hf_')"
    echo ""

    while true; do
        local token=$(get_user_input "Enter your Hugging Face token" "" validate_hf_token)
        if [ -n "$token" ]; then
            echo "$token"
            return 0
        fi
    done
}

# Function to create terraform.tfvars
create_tfvars() {
    local tfvars_file="terraform.tfvars"

    print_header "Configuration"
    echo "=============="

    # Get user inputs
    local region=$(get_user_input "AWS region" "us-east-1" validate_region)
    local instance_type=$(get_user_input "Instance type" "g4dn.xlarge" validate_instance_type)
    local spot_price=$(get_user_input "Maximum spot price per hour" "0.50" validate_spot_price)
    local volume_size=$(get_user_input "Volume size (GB)" "100")
    local ssh_key_path=$(get_user_input "SSH key path" "~/.ssh/id_rsa")
    local hf_token=$(get_hf_token)
    local wandb_api_key=$(get_user_input "W&B API key (optional -- blank runs offline)" "")
    local train_image=$(get_user_input "Training image (e.g. roshbeed/ai-residency-vit-classifier-train:latest -- push it first with: make push SERVICE=<name>)" "" validate_train_image)
    local train_command=$(get_user_input "Override the image's default training command (optional, e.g. 'seq2seq')" "")
    local max_training_time=$(get_user_input "Maximum training time (hours)" "24")

    # Create terraform.tfvars
    cat > "$tfvars_file" << EOF
# AWS Configuration
aws_region = "$region"
instance_type = "$instance_type"
spot_price = "$spot_price"
volume_size = $volume_size

# SSH Configuration
ssh_key_path = "$ssh_key_path"

# Credentials -- training pushes its own checkpoint and logs, nothing here re-uploads it
hf_token = "$hf_token"
wandb_api_key = "$wandb_api_key"

# Which week to train
train_image = "$train_image"
train_command = "$train_command"

# Training Configuration
max_training_time = $max_training_time

# Environment
environment = "training"
EOF

    print_status "Configuration saved to $tfvars_file"
    echo ""
}

# Function to deploy infrastructure
deploy_infrastructure() {
    print_header "Deploying Infrastructure"
    echo "==========================="
    
    print_status "Initializing Terraform..."
    terraform init
    
    print_status "Planning deployment..."
    terraform plan
    
    echo ""
    read -p "Deploy infrastructure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Deploying infrastructure..."
        terraform apply -auto-approve
        
        print_status "Infrastructure deployed successfully!"
        echo ""
        
        # Show outputs
        print_header "Deployment Summary"
        echo "==================="
        terraform output
        
        echo ""
        print_status "Next steps:"
        echo "1. Monitor training: ./monitor.sh"
        echo "2. View logs: ./monitor.sh --logs"
        echo "3. Check GPU: ./monitor.sh --gpu"
        echo "4. Cleanup when done: ./cleanup.sh"
        
    else
        print_warning "Deployment cancelled."
        exit 0
    fi
}

# Main function
main() {
    # Check if we're in the infra directory
    if [ ! -f "main.tf" ]; then
        print_error "Please run this script from the infra directory."
        exit 1
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Setup SSH key
    setup_ssh_key "~/.ssh/id_rsa"
    
    # Create configuration
    create_tfvars
    
    # Deploy infrastructure
    deploy_infrastructure
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo ""
        echo "This script will:"
        echo "  1. Check prerequisites (Terraform, AWS CLI, credentials)"
        echo "  2. Setup SSH key if needed"
        echo "  3. Get configuration from user input"
        echo "  4. Deploy infrastructure"
        echo "  5. Show monitoring commands"
        echo ""
        echo "Prerequisites:"
        echo "  - Terraform installed"
        echo "  - AWS CLI installed and configured"
        echo "  - Run from infra directory"
        exit 0
        ;;
    "")
        # No arguments, run normally
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information."
        exit 1
        ;;
esac

# Run main function
main 