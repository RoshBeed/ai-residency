# AWS spot GPU training — an alternative to RunPod

Terraform that provisions a single AWS spot GPU instance, pulls a pushed `roshbeed/ai-residency-weekN-train`
image, runs it, and terminates the instance itself when training finishes (or times out). This is
the same training container every week repo's own `Dockerfile.train` already builds and
`make push SERVICE=<name>` already pushes — this module doesn't reimplement training, it's just
another place to run the same image, on a rented GPU instead of RunPod.

Generalized from a module that only ever ran the vision-transformers project,
by cloning its git repo and building a Python environment by hand on the instance. That approach
predates this portfolio's move to Docker images with a shared base — now the instance just needs
Docker and the NVIDIA Container Toolkit, and `train_image` picks which week's already-built image to
pull and run.

**Found and fixed while moving it, not new bugs introduced by the move:** the original
`user_data.sh` referenced three Terraform template variables (`training_type`, `max_training_time`,
and — inside a bash function — `timeout_hours`) that were never actually passed to the
`templatefile()` call in `main.tf`. `main.tf` and `outputs.tf` also both defined the same four
outputs (`instance_id`, `public_ip`, `instance_type`, `spot_price`), which Terraform rejects
outright. Either bug alone means this module could never have passed `terraform validate`, let alone
been applied — it was never actually run. Both are fixed here (see `user_data.sh`'s and `main.tf`'s
own comments for what changed).

## Prerequisites

- Terraform >= 1.0, AWS CLI configured (`aws configure`), an SSH key pair
- A pushed training image: `cd ai-residency-weekN-.../ && make push SERVICE=<name>` (needs `docker login`
  first if the image is private — see `docker_hub_username`/`docker_hub_token` below)
- A Hugging Face token with write access — training pushes its own checkpoint as it runs

## Usage

```bash
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: at minimum, train_image and hf_token

terraform init
terraform plan     # review before spending anything
terraform apply    # provisions a real, billable spot instance
```

Or run `./quick-start.sh` for an interactive prompt-driven setup that writes `terraform.tfvars` for
you. `./monitor.sh` shows instance status, GPU utilization, and setup/training logs over SSH.
`./cleanup.sh` tears the instance down (also happens automatically on training completion or
timeout — this is a backstop, not the normal path).

## Variables worth knowing

| Variable | What it does |
|---|---|
| `train_image` | The image to pull and run, e.g. `roshb/ai-residency-vit-classifier-train:latest`. No default — always set this. |
| `train_command` | Overrides the image's own default training entrypoint. Empty runs the image's default (e.g. Week 3's classifier stage); set to `"seq2seq"` for its other trainer. |
| `docker_hub_username` / `docker_hub_token` | Only needed if `train_image` is a private repository. |
| `hf_token` | Training pushes its own checkpoint to the Hub via `libs/common`'s `push_checkpoint` — this module never re-uploads anything separately. |
| `wandb_api_key` | Optional. Without it, the run still trains and logs offline (`libs/common`'s `init_tracking` degrades rather than fails). |
| `spot_price` / `max_training_time` | Caps cost two ways: a max hourly bid, and an auto-terminate timeout regardless of training progress. |

## What actually happens on the instance

`user_data.sh` runs once, at boot, as root:

1. Installs Docker and the NVIDIA Container Toolkit, verifies GPU access (`nvidia-smi`, then a real
   `docker run --gpus all` smoke test).
2. `docker pull`s `train_image` (optionally `docker login`s first, if given credentials for a
   private image).
3. Runs it with `--gpus all`, `HF_TOKEN`/`WANDB_API_KEY` as environment, `train_command` as an
   override if set, under a timeout of `max_training_time` hours.
4. Terminates the instance itself — on success, on timeout, or on failure. A spot instance left
   idle after training finishes is a real, avoidable cost; this closes that gap regardless of how
   the run ends.

No log file is written to a fixed path; `/var/log/cloud-init-output.log` (the standard place
cloud-init captures a boot script's own output) has the setup steps, and `docker logs` on the
training container has its own output — `monitor.sh` reads both.
