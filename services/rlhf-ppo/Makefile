.DEFAULT_GOAL := help
.PHONY: train-loop train serve test help

train:  ## Run ONE PPO step on two example prompts
	uv run python -m training.run_ppo

train-loop:  ## Run a PPO loop on real TL;DR prompts, reward measured before and after
	uv run python -m rlhf_ppo.training.run_ppo_loop

serve:  ## Run the FastAPI generation service on :8000 (no trained PPO checkpoint — see README)
	uv run python -m serving.api

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
