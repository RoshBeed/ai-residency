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

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${var.ssh_key_path} ubuntu@${aws_spot_instance_request.training_instance.public_ip}"
}

output "training_logs_command" {
  description = "Command to view training logs"
  value       = "ssh -i ${var.ssh_key_path} ubuntu@${aws_spot_instance_request.training_instance.public_ip} 'tail -f /opt/transformer-training/training.log'"
}

output "estimated_cost_per_hour" {
  description = "Estimated cost per hour based on spot price"
  value       = var.spot_price
}

output "max_total_cost" {
  description = "Maximum total cost for the training session"
  value       = "${var.spot_price} * ${var.max_training_time} hours = $${parsefloat(var.spot_price) * var.max_training_time}"
}

output "huggingface_repo" {
  description = "Hugging Face repository where models will be uploaded"
  value       = "https://huggingface.co/${var.hf_repo}"
} 