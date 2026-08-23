.DEFAULT_GOAL := help
.PHONY: install train train-corpus train-cbow train-cbow-sample \
        sweep visualise serve test \
        fmt lint typecheck check \
        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

install: ## Create a uv-managed venv and install dependencies — never touches the host/system Python
	uv sync

## --- Training (2 pipeline stages) ---
## All commands run from the repo root: paths in Config (data/, checkpoints/) are
## relative to it. These local targets are the fast path on Apple Silicon, since
## get_device() selects MPS when available. `make docker-train` is always CPU-only
## (Docker Desktop's Linux VM has no MPS passthrough) — it exists for portability
## and a future real GPU host, not for speed on this machine.

train-corpus: ## Stage 1: build the merged vocabulary/corpus
	uv run python -m word_embeddings.etl

train-cbow: ## Stage 2: pretrain CBOW embeddings (hours on the real corpus)
	uv run python -m word_embeddings.train

train-cbow-sample: ## Stage 2 against a tiny built-in corpus — seconds, writes a separate checkpoint
	uv run python -m word_embeddings.train --sample

# WARNING: multi-hour end to end — corpus download+lemmatize, then CBOW over ~100M
# tokens, run back to back. Shorten either stage with the AI_RESIDENCY_* config
# overrides, e.g. AI_RESIDENCY_CBOW_NUM_EPOCHS=1.
train: train-corpus train-cbow ## Run both training stages in sequence (multi-hour, see warning above)

sweep: ## Run a W&B sweep agent: make sweep SWEEP=<id from `wandb sweep sweep.yaml`>
	@test -n "$(SWEEP)" || (echo "Usage: make sweep SWEEP=<sweep id>   (create one with: wandb sweep sweep.yaml)" && exit 1)
	uv run wandb agent $(SWEEP)

visualise: ## Write a t-SNE plot of the learned embedding space to assets/
	uv run python -m word_embeddings.evaluate --visualise

serve: ## Run the FastAPI nearest-neighbour/analogy service locally (no Docker)
	uv run python -m serving.api

test: ## Run the test suite
	uv run pytest -q

fmt: ## Format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Check formatting and lint rules without changing anything
	uv run ruff format --check .
	uv run ruff check .

typecheck: ## Run mypy in strict mode
	uv run mypy config serving word_embeddings tests

check: lint typecheck test ## Everything CI would run

docker-build-train: ## Build the training image (FROM roshbeed/ai-residency-base — build that first)
	docker build -f Dockerfile.train -t roshbeed/ai-residency-week1-train:latest .

docker-build-serve: ## Build the serving (inference) image (FROM roshbeed/ai-residency-base)
	docker build -f Dockerfile.serve -t roshbeed/ai-residency-week1-serve:latest .

docker-push-train: ## Push the training image (for a RunPod/AWS GPU host to pull)
	docker push roshbeed/ai-residency-week1-train:latest

docker-push-serve: ## Push the serving image (for Kubernetes to pull)
	docker push roshbeed/ai-residency-week1-serve:latest

docker-up: ## Start the serve service
	docker compose up --build serve

docker-train: ## Run the train container once, profile-gated (never runs on a bare `up`)
	docker compose --profile train run --rm train

docker-down: ## Stop and remove all containers started by docker compose
	docker compose down

clean: ## Remove local venvs/caches/offline wandb runs (NOT data/models/artifacts)
	# Deliberately does NOT remove data/, checkpoints/ or assets/ — those are real,
	# expensive-to-regenerate training artifacts, not build/cache junk.
	rm -rf .venv .venv-run .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -not -path "./data/*" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf wandb

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
