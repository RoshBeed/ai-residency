.PHONY: install train serve test fmt lint typecheck check \
        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

install:  ## Install the package and dev dependencies
	uv sync

# Device: PPOTrainer picks cuda > mps > cpu via ai_residency_common.get_device(). On Apple
# Silicon this runs on MPS; inside Docker (Linux container, no MPS passthrough) it is
# always CPU regardless of host.
train:  ## Run ONE PPO step on two example prompts (this repo has no multi-epoch loop)
	uv run python -m training.run_ppo

serve:  ## Run the FastAPI generation service on :8000 (no trained PPO checkpoint — see README)
	uv run python -m serving.api

test:  ## Run the test suite
	uv run pytest -q

fmt:  ## Format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Check formatting and lint rules without changing anything
	uv run ruff format --check .
	uv run ruff check .

typecheck:  ## Run mypy in strict mode
	uv run mypy sa_pair advantage base_setup config experience_buffer models ppo_trainer reward_kl token_attribution training serving tests

check: lint typecheck test  ## Everything CI would run

docker-build-train:  ## Build the training image (FROM roshbeed/ai-residency-base — build that first)
	docker build -f Dockerfile.train -t roshbeed/ai-residency-week6-train:latest .

docker-build-serve:  ## Build the serving image — the real FastAPI generation service (FROM roshbeed/ai-residency-base)
	docker build -f Dockerfile.serve -t roshbeed/ai-residency-week6-serve:latest .

docker-push-train:  ## Push the training image (for a RunPod/AWS GPU host to pull)
	docker push roshbeed/ai-residency-week6-train:latest

docker-push-serve:  ## Push the serving image (for Kubernetes to pull)
	docker push roshbeed/ai-residency-week6-serve:latest

docker-up:  ## Start the serve service (the real FastAPI generation service)
	docker compose up --build serve

docker-train:  ## Run one PPO step in a container (profile-gated)
	docker compose --profile train run --rm train

docker-down:  ## Stop everything and remove volumes
	docker compose down -v

clean:  ## Remove caches and build artifacts
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
