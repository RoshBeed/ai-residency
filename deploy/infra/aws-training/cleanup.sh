#!/bin/bash

# AI Residency Portfolio AWS Training Infrastructure Cleanup Script

set -e

echo "🧹 Starting cleanup of the AI Residency training infrastructure..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    print_error "Terraform is not installed. Please install Terraform first."
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install AWS CLI first."
    exit 1
fi

# Check if we're in the infra directory
if [ ! -f "main.tf" ]; then
    print_error "Please run this script from the infra directory."
    exit 1
fi

# Function to check if infrastructure exists
check_infrastructure() {
    if terraform state list 2>/dev/null | grep -q "aws_spot_instance_request"; then
        return 0
    else
        return 1
    fi
}

# Function to get instance ID
get_instance_id() {
    terraform output -raw instance_id 2>/dev/null || echo ""
}

# Function to check instance status
check_instance_status() {
    local instance_id=$1
    if [ -n "$instance_id" ]; then
        aws ec2 describe-instances --instance-ids "$instance_id" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

# Main cleanup process
main() {
    print_status "Checking current infrastructure state..."
    
    if ! check_infrastructure; then
        print_warning "No infrastructure found to clean up."
        print_status "Cleaning up local Terraform files..."
        rm -rf .terraform .terraform.lock.hcl terraform.tfstate*
        print_status "Cleanup complete!"
        exit 0
    fi
    
    # Get instance details
    INSTANCE_ID=$(get_instance_id)
    if [ -n "$INSTANCE_ID" ]; then
        INSTANCE_STATUS=$(check_instance_status "$INSTANCE_ID")
        print_status "Found instance: $INSTANCE_ID (Status: $INSTANCE_STATUS)"
        
        if [ "$INSTANCE_STATUS" = "running" ]; then
            print_warning "Instance is still running. This will terminate it."
            read -p "Do you want to continue? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_status "Cleanup cancelled."
                exit 0
            fi
        fi
    fi
    
    # Destroy infrastructure
    print_status "Destroying infrastructure..."
    terraform destroy -auto-approve
    
    if [ $? -eq 0 ]; then
        print_status "Infrastructure destroyed successfully!"
    else
        print_error "Failed to destroy infrastructure. Please check the error messages above."
        exit 1
    fi
    
    # Clean up local files
    print_status "Cleaning up local Terraform files..."
    rm -rf .terraform .terraform.lock.hcl terraform.tfstate*
    
    print_status "Cleanup complete! 🎉"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --force       Skip confirmation prompts"
        echo ""
        echo "This script will:"
        echo "  1. Check if infrastructure exists"
        echo "  2. Destroy all AWS resources"
        echo "  3. Clean up local Terraform files"
        echo ""
        echo "Prerequisites:"
        echo "  - Terraform installed"
        echo "  - AWS CLI configured"
        echo "  - Run from infra directory"
        exit 0
        ;;
    --force)
        # Skip confirmation for automated scripts
        FORCE=true
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

# Run main cleanup
main 