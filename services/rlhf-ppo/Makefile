        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

train:  ## Run ONE PPO step on two example prompts (this repo has no multi-epoch loop)
	uv run python -m training.run_ppo

serve:  ## Run the FastAPI generation service on :8000 (no trained PPO checkpoint — see README)
	uv run python -m serving.api

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
