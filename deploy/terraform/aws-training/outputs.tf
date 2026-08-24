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
  description = "Spot price cap for the instance. Empty when bidding the on-demand price, which is the default."
  value       = aws_spot_instance_request.training_instance.spot_price
}

output "session_command" {
  description = "How to get a shell on the instance. Uses SSM Session Manager, which needs no open port and no key -- the instance role carries AmazonSSMManagedInstanceCore for exactly this."
  value       = "aws ssm start-session --target ${aws_spot_instance_request.training_instance.spot_instance_id}"
}

output "ssh_command" {
  description = "SSH alternative, usable only when allowed_ssh_cidr and ssh_key_name were both set. Empty otherwise, since with no inbound rule there is nothing to connect to."
  value       = var.allowed_ssh_cidr == "" ? "" : "ssh -i ${var.ssh_key_path} ubuntu@${aws_spot_instance_request.training_instance.public_ip}"
}

output "training_logs_command" {
  description = "How to follow the training run. There is no separate log file -- cloud-init captures everything user_data.sh printed, and once training starts `docker logs` follows the container itself."
  value       = "aws ssm start-session --target ${aws_spot_instance_request.training_instance.spot_instance_id} --document-name AWS-StartInteractiveCommand --parameters command='tail -f /var/log/cloud-init-output.log'"
}

output "estimated_cost_per_hour" {
  description = "Cost ceiling per hour, when one was set. Empty means the run bids the on-demand price and pays the current spot rate, so the ceiling is the on-demand price for this instance type."
  value       = var.spot_price
}

output "max_total_cost" {
  # The original version of this output used "$${...}", which is Terraform's escape
  # for a *literal* "${" -- it never actually evaluated the expression at all, just
  # printed it back out as text. It also called a "parsefloat" function that does not
  # exist in Terraform (the real one is tonumber()). format() sidesteps the escaping
  # ambiguity by keeping the literal "$" inside the format string's own argument.
  description = "Maximum total cost for the training session, when a spot_price ceiling was set."
  value       = var.spot_price == "" ? "no ceiling set; bidding the on-demand price" : "${format("$%.2f", tonumber(var.spot_price) * var.max_training_time)} (${var.spot_price}/hr * ${var.max_training_time}h)"
}

output "train_image" {
  description = "The training image this run pulled and ran"
  value       = var.train_image
}
