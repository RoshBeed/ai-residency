.DEFAULT_GOAL := help
.PHONY: train-loop train serve test help

# Both targets pin the device to CPU, and that is not a typo on an Apple Silicon
# machine. A PPO step holds activations for four model roles across twelve stages, and
# on MPS that accumulates past the backend's high-watermark ceiling and dies mid-step
# with `MPS backend out of memory (MPS allocated: 42.45 GiB, max allowed: 42.43 GiB)`
# -- after spending 252-413s in the value update alone. The same step on CPU completes
# in 70s with every stage finite: value update 17.3s, policy update 22.0s. CPU is both
# the faster and the only working device here, so the service says so rather than
# leaving each caller to find out.
DEVICE ?= cpu

train:  ## Run ONE PPO step on two example prompts
	AI_RESIDENCY_DEVICE=$(DEVICE) uv run python -m rlhf_ppo.training.run_ppo

train-loop:  ## Run a PPO loop on real TL;DR prompts, reward measured before and after
	AI_RESIDENCY_DEVICE=$(DEVICE) uv run python -m rlhf_ppo.training.run_ppo_loop

serve:  ## Run the FastAPI generation service on :8000 (no trained PPO checkpoint — see README)
	uv run python -m serving.api

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
