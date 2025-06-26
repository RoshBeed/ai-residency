#!/bin/bash

# Transformer Training Infrastructure Monitoring Script

set -e

echo "📊 Transformer Training Monitor"

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

# Function to get instance details
get_instance_details() {
    local instance_id=$(terraform output -raw instance_id 2>/dev/null || echo "")
    local public_ip=$(terraform output -raw public_ip 2>/dev/null || echo "")
    local instance_type=$(terraform output -raw instance_type 2>/dev/null || echo "")
    local spot_price=$(terraform output -raw spot_price 2>/dev/null || echo "")
    
    echo "$instance_id|$public_ip|$instance_type|$spot_price"
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

# Function to get instance launch time
get_launch_time() {
    local instance_id=$1
    if [ -n "$instance_id" ]; then
        aws ec2 describe-instances --instance-ids "$instance_id" --query 'Reservations[0].Instances[0].LaunchTime' --output text 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

# Function to calculate uptime
calculate_uptime() {
    local launch_time=$1
    if [ "$launch_time" != "unknown" ]; then
        local launch_epoch=$(date -d "$launch_time" +%s)
        local current_epoch=$(date +%s)
        local uptime_seconds=$((current_epoch - launch_epoch))
        local hours=$((uptime_seconds / 3600))
        local minutes=$(((uptime_seconds % 3600) / 60))
        echo "${hours}h ${minutes}m"
    else
        echo "unknown"
    fi
}

# Function to estimate cost
estimate_cost() {
    local uptime=$1
    local spot_price=$2
    
    if [ "$uptime" != "unknown" ] && [ -n "$spot_price" ]; then
        local hours=$(echo "$uptime" | grep -o '[0-9]*h' | grep -o '[0-9]*')
        if [ -n "$hours" ]; then
            local cost=$(echo "$hours * $spot_price" | bc -l 2>/dev/null || echo "0")
            printf "%.2f" "$cost"
        else
            echo "0.00"
        fi
    else
        echo "0.00"
    fi
}

# Function to check if SSH is available
check_ssh_availability() {
    local public_ip=$1
    local ssh_key_path=$(terraform output -raw ssh_key_path 2>/dev/null || echo "~/.ssh/id_rsa")
    
    if [ -n "$public_ip" ]; then
        timeout 5 ssh -i "$ssh_key_path" -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@"$public_ip" "echo 'SSH OK'" 2>/dev/null && echo "available" || echo "unavailable"
    else
        echo "unavailable"
    fi
}

# Function to get training logs
get_training_logs() {
    local public_ip=$1
    local ssh_key_path=$(terraform output -raw ssh_key_path 2>/dev/null || echo "~/.ssh/id_rsa")
    
    if [ -n "$public_ip" ]; then
        ssh -i "$ssh_key_path" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@"$public_ip" "tail -n 50 /opt/transformer-training/training.log 2>/dev/null || echo 'No training logs found'" 2>/dev/null || echo "Unable to fetch logs"
    else
        echo "No instance IP available"
    fi
}

# Function to check GPU status
get_gpu_status() {
    local public_ip=$1
    local ssh_key_path=$(terraform output -raw ssh_key_path 2>/dev/null || echo "~/.ssh/id_rsa")
    
    if [ -n "$public_ip" ]; then
        ssh -i "$ssh_key_path" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@"$public_ip" "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 'GPU not available'" 2>/dev/null || echo "Unable to fetch GPU status"
    else
        echo "No instance IP available"
    fi
}

# Function to check system resources
get_system_resources() {
    local public_ip=$1
    local ssh_key_path=$(terraform output -raw ssh_key_path 2>/dev/null || echo "~/.ssh/id_rsa")
    
    if [ -n "$public_ip" ]; then
        ssh -i "$ssh_key_path" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@"$public_ip" "free -h | grep '^Mem:' | awk '{print \"Memory: \" \$3 \"/\" \$2}' && df -h / | tail -1 | awk '{print \"Disk: \" \$3 \"/\" \$2}'" 2>/dev/null || echo "Unable to fetch system resources"
    else
        echo "No instance IP available"
    fi
}

# Main monitoring function
main() {
    print_header "Infrastructure Status"
    echo "========================"
    
    # Get instance details
    local details=$(get_instance_details)
    local instance_id=$(echo "$details" | cut -d'|' -f1)
    local public_ip=$(echo "$details" | cut -d'|' -f2)
    local instance_type=$(echo "$details" | cut -d'|' -f3)
    local spot_price=$(echo "$details" | cut -d'|' -f4)
    
    if [ -z "$instance_id" ]; then
        print_warning "No infrastructure found. Run 'terraform apply' first."
        exit 0
    fi
    
    # Get status information
    local status=$(check_instance_status "$instance_id")
    local launch_time=$(get_launch_time "$instance_id")
    local uptime=$(calculate_uptime "$launch_time")
    local cost=$(estimate_cost "$uptime" "$spot_price")
    local ssh_status=$(check_ssh_availability "$public_ip")
    
    # Display instance information
    echo "Instance ID:     $instance_id"
    echo "Public IP:       $public_ip"
    echo "Instance Type:   $instance_type"
    echo "Spot Price:      \$$spot_price/hour"
    echo "Status:          $status"
    echo "Launch Time:     $launch_time"
    echo "Uptime:          $uptime"
    echo "Estimated Cost:  \$$cost"
    echo "SSH Access:      $ssh_status"
    echo ""
    
    # Show detailed information if instance is running
    if [ "$status" = "running" ]; then
        print_header "System Resources"
        echo "=================="
        get_system_resources "$public_ip"
        echo ""
        
        print_header "GPU Status"
        echo "==========="
        get_gpu_status "$public_ip"
        echo ""
        
        print_header "Recent Training Logs"
        echo "======================"
        get_training_logs "$public_ip"
        echo ""
        
        # Show useful commands
        print_header "Useful Commands"
        echo "================"
        echo "SSH into instance:"
        echo "  ssh -i $(terraform output -raw ssh_key_path) ubuntu@$public_ip"
        echo ""
        echo "View live training logs:"
        echo "  ssh -i $(terraform output -raw ssh_key_path) ubuntu@$public_ip 'tail -f /opt/transformer-training/training.log'"
        echo ""
        echo "Check GPU usage:"
        echo "  ssh -i $(terraform output -raw ssh_key_path) ubuntu@$public_ip 'nvidia-smi'"
        echo ""
        echo "Monitor system resources:"
        echo "  ssh -i $(terraform output -raw ssh_key_path) ubuntu@$public_ip 'htop'"
        echo ""
    fi
    
    # Show cost warning if running for a long time
    if [ "$status" = "running" ] && [ "$cost" != "0.00" ]; then
        local cost_float=$(echo "$cost" | bc -l 2>/dev/null || echo "0")
        if (( $(echo "$cost_float > 10" | bc -l) )); then
            print_warning "Instance has been running for a while. Current cost: \$$cost"
            print_warning "Consider checking if training is complete or if cleanup is needed."
        fi
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --logs        Show only training logs"
        echo "  --gpu         Show only GPU status"
        echo "  --resources   Show only system resources"
        echo ""
        echo "This script will:"
        echo "  1. Check infrastructure status"
        echo "  2. Display instance information"
        echo "  3. Show system resources and GPU status"
        echo "  4. Display recent training logs"
        echo "  5. Provide useful SSH commands"
        echo ""
        echo "Prerequisites:"
        echo "  - Terraform installed"
        echo "  - AWS CLI configured"
        echo "  - Run from infra directory"
        exit 0
        ;;
    --logs)
        details=$(get_instance_details)
        public_ip=$(echo "$details" | cut -d'|' -f2)
        print_header "Training Logs"
        echo "=============="
        get_training_logs "$public_ip"
        exit 0
        ;;
    --gpu)
        details=$(get_instance_details)
        public_ip=$(echo "$details" | cut -d'|' -f2)
        print_header "GPU Status"
        echo "==========="
        get_gpu_status "$public_ip"
        exit 0
        ;;
    --resources)
        details=$(get_instance_details)
        public_ip=$(echo "$details" | cut -d'|' -f2)
        print_header "System Resources"
        echo "=================="
        get_system_resources "$public_ip"
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

# Run main monitoring
main 