.PHONY: install train train-classifier train-seq2seq serve test fmt lint typecheck check \
        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

# ai-residency-common is a private repo, so only building the shared base image needs a
# token — `gh auth token` supplies it without it ever being stored in a file.
# Every image below builds FROM that base and never touches git.
GITHUB_TOKEN ?= $(shell gh auth token)
export GITHUB_TOKEN

install:  ## Install the package and dev dependencies
	uv sync

train: train-classifier  ## Alias for the faster of the two training runs

train-classifier:  ## Train the single-digit classifier
	uv run python -m models.classifier.trainer

train-seq2seq:  ## Train the multi-digit seq2seq model
	uv run python -m models.seq2seq.trainer

serve:  ## Run the FastAPI backend on :8000 (loads both committed checkpoints)
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
	uv run mypy config models serving transformer datasets tests

check: lint typecheck test  ## Everything CI would run

# Note (Apple Silicon): Docker Desktop's Linux VM has no MPS passthrough, so every
# docker-* target runs on CPU regardless of host hardware. Use the native targets above
# for real M-series acceleration.

docker-build-train:  ## Build the training image
	docker build -f Dockerfile.train -t roshbeed/ai-residency-week3-train:latest .

docker-build-serve:  ## Build the serving (inference) image
	docker build -f Dockerfile.serve -t roshbeed/ai-residency-week3-serve:latest .

docker-push-train:  ## Push the training image (for a RunPod/AWS GPU host to pull)
	docker push roshbeed/ai-residency-week3-train:latest

docker-push-serve:  ## Push the serving image (for Kubernetes to pull)
	docker push roshbeed/ai-residency-week3-serve:latest

docker-up:  ## Start the serve service
	docker compose up --build serve

docker-train:  ## Train the ViT classifier in the train container (override CMD for seq2seq)
	docker compose --profile train run --rm train

docker-down:  ## Stop everything and remove volumes
	docker compose down -v

clean:  ## Remove caches and build artifacts (never touches checkpoints/ or data/)
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
