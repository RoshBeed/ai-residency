#!/bin/bash

# AI Residency Portfolio AWS Training Infrastructure Monitoring Script

set -e

echo "📊 AI Residency Training Monitor"

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

# Run a command on the instance through SSM, not SSH.
#
# The instance has no inbound rules by default -- allowed_ssh_cidr is empty -- so there
# is no port to connect to. SSM reaches it the other way round: the agent on the box
# polls AWS, and the instance role carries AmazonSSMManagedInstanceCore for exactly
# this. send-command is used rather than start-session because it is non-interactive
# and returns the output as data.
remote_exec() {
    local instance_id=$1
    local script=$2

    if [ -z "$instance_id" ]; then
        echo "No instance available"
        return
    fi

    local command_id
    command_id=$(aws ssm send-command \
        --instance-ids "$instance_id" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"$script\"]" \
        --query 'Command.CommandId' --output text 2>/dev/null || echo "")

    if [ -z "$command_id" ]; then
        echo "Unable to reach the instance over SSM (is the agent registered yet?)"
        return
    fi

    aws ssm wait command-executed --command-id "$command_id" --instance-id "$instance_id" 2>/dev/null || true
    aws ssm get-command-invocation \
        --command-id "$command_id" --instance-id "$instance_id" \
        --query 'StandardOutputContent' --output text 2>/dev/null || echo "Unable to fetch output"
}

# Function to check whether the instance is reachable at all
check_agent_availability() {
    local instance_id=$1

    if [ -z "$instance_id" ]; then
        echo "unavailable"
        return
    fi

    local ping_status
    ping_status=$(aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$instance_id" \
        --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || echo "")

    if [ "$ping_status" = "Online" ]; then echo "available"; else echo "unavailable"; fi
}

# Function to get training logs
get_training_logs() {
    # No separate log file: cloud-init captures user_data.sh's own output (setup,
    # docker pull, the run_with_timeout wrapper's log lines) here; the training
    # container's own stdout is in `docker logs` once it is running.
    remote_exec "$1" "tail -n 50 /var/log/cloud-init-output.log 2>/dev/null || echo 'No setup logs found'; echo '---'; docker logs --tail 50 \$(docker ps -lq) 2>/dev/null || echo 'No training container logs found'"
}

# Function to check GPU status
get_gpu_status() {
    remote_exec "$1" "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 'GPU not available'"
}

# Function to check system resources
get_system_resources() {
    remote_exec "$1" "free -h | grep '^Mem:' | awk '{print \"Memory: \" \$3 \"/\" \$2}'; df -h / | tail -1 | awk '{print \"Disk: \" \$3 \"/\" \$2}'"
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
    local agent_status=$(check_agent_availability "$instance_id")
    
    # Display instance information
    echo "Instance ID:     $instance_id"
    echo "Public IP:       $public_ip"
    echo "Instance Type:   $instance_type"
    echo "Spot Price:      \$$spot_price/hour"
    echo "Status:          $status"
    echo "Launch Time:     $launch_time"
    echo "Uptime:          $uptime"
    echo "Estimated Cost:  \$$cost"
    echo "SSH Access:      $agent_status"
    echo ""
    
    # Show detailed information if instance is running
    if [ "$status" = "running" ]; then
        print_header "System Resources"
        echo "=================="
        get_system_resources "$instance_id"
        echo ""
        
        print_header "GPU Status"
        echo "==========="
        get_gpu_status "$instance_id"
        echo ""
        
        print_header "Recent Training Logs"
        echo "======================"
        get_training_logs "$instance_id"
        echo ""
        
        # Show useful commands
        print_header "Useful Commands"
        echo "================"
        echo "Shell on the instance (no open port needed):"
        echo "  aws ssm start-session --target $instance_id"
        echo ""
        echo "View live training logs:"
        echo "  aws ssm start-session --target $instance_id \\"
        echo "    --document-name AWS-StartInteractiveCommand \\"
        echo "    --parameters command='tail -f /var/log/cloud-init-output.log'"
        echo ""
        echo "Check GPU usage:"
        echo "  aws ssm start-session --target $instance_id \\"
        echo "    --document-name AWS-StartInteractiveCommand --parameters command='nvidia-smi'"
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
        get_training_logs "$instance_id"
        exit 0
        ;;
    --gpu)
        details=$(get_instance_details)
        public_ip=$(echo "$details" | cut -d'|' -f2)
        print_header "GPU Status"
        echo "==========="
        get_gpu_status "$instance_id"
        exit 0
        ;;
    --resources)
        details=$(get_instance_details)
        public_ip=$(echo "$details" | cut -d'|' -f2)
        print_header "System Resources"
        echo "=================="
        get_system_resources "$instance_id"
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