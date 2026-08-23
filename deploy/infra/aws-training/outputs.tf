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
  description = "Command to view training logs. There's no separate log file -- cloud-init captures everything user_data.sh printed here, and once training starts, `docker logs` follows the container itself."
  value       = "ssh -i ${var.ssh_key_path} ubuntu@${aws_spot_instance_request.training_instance.public_ip} 'tail -f /var/log/cloud-init-output.log'"
}

output "estimated_cost_per_hour" {
  description = "Estimated cost per hour based on spot price"
  value       = var.spot_price
}

output "max_total_cost" {
  # The original version of this output used "$${...}", which is Terraform's escape
  # for a *literal* "${" -- it never actually evaluated the expression at all, just
  # printed it back out as text. It also called a "parsefloat" function that does not
  # exist in Terraform (the real one is tonumber()). format() sidesteps the escaping
  # ambiguity by keeping the literal "$" inside the format string's own argument.
  description = "Maximum total cost for the training session"
  value       = "${format("$%.2f", tonumber(var.spot_price) * var.max_training_time)} (${var.spot_price}/hr * ${var.max_training_time}h)"
}

output "train_image" {
  description = "The training image this run pulled and ran"
  value       = var.train_image
}
