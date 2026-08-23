.PHONY: install train serve zero-shot test fmt lint typecheck check \
        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

install:  ## Install the package and dev dependencies
	uv sync

train:  ## Train the CLIP-to-Qwen3 adapter on Flickr30k
	uv run python -m multimodal_captioning.training

serve:  ## Run the FastAPI backend on :8000
	uv run python -m serving.api

zero-shot:  ## Run Part 1: CLIP zero-shot classification over data/images/ (exits when done)
	uv run python -m multimodal_captioning.zero_shot

test:  ## Run the test suite
	uv run pytest -q

fmt:  ## Format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Check formatting and lint rules without changing anything
	uv run ruff format --check .
	uv run ruff check .

typecheck:  ## Run mypy in strict mode
	uv run mypy config multimodal_captioning serving tests

check: lint typecheck test  ## Everything CI would run

# Note (Apple Silicon): Docker Desktop's Linux VM has no MPS passthrough, so every
# docker-* target runs on CPU regardless of host hardware. Use the native targets above
# for real M-series acceleration.

docker-build-train:  ## Build the training image (FROM roshbeed/ai-residency-base — build that first)
	docker build -f Dockerfile.train -t roshbeed/ai-residency-week4-train:latest .

docker-build-serve:  ## Build the serving (inference) image (FROM roshbeed/ai-residency-base)
	docker build -f Dockerfile.serve -t roshbeed/ai-residency-week4-serve:latest .

docker-push-train:  ## Push the training image (for a RunPod/AWS GPU host to pull)
	docker push roshbeed/ai-residency-week4-train:latest

docker-push-serve:  ## Push the serving image (for Kubernetes to pull)
	docker push roshbeed/ai-residency-week4-serve:latest

docker-up:  ## Start the serve service
	docker compose up --build serve

docker-train:  ## Run training in a container (profile-gated)
	docker compose --profile train run --rm train

docker-down:  ## Stop everything and remove volumes
	docker compose down -v

clean:  ## Remove caches and build artifacts (never touches checkpoints/ or data/)
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
