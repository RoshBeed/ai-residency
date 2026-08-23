.DEFAULT_GOAL := help
.PHONY: install train predict serve test \
        fmt lint typecheck check \
        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

install: ## Create a uv-managed venv and install dependencies — never touches the host/system Python
	uv sync

## --- Training ---
## All commands run from the repo root. The frozen CBOW embeddings this stage consumes
## are pulled from the word-embeddings repo's Hugging Face repo (or a local copy at
## checkpoints/cbow/). Shorten the run with AI_RESIDENCY_* overrides, e.g.
## AI_RESIDENCY_NUM_EPOCHS=1.

train: ## Train the regression head over the frozen CBOW embeddings
	uv run python -m content_scoring.train

predict: ## Score a title from the command line: make predict TITLE="..."
	@test -n "$(TITLE)" || (echo "Usage: make predict TITLE='your title here'" && exit 1)
	uv run python -m content_scoring.predict "$(TITLE)"

serve: ## Run the FastAPI prediction service locally (no Docker)
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
	uv run mypy config serving content_scoring tests

check: lint typecheck test ## Everything CI would run

docker-build-train: ## Build the training image (FROM roshbeed/ai-residency-base — build that first)
	docker build -f Dockerfile.train -t roshbeed/ai-residency-week1-content-scoring-train:latest .

docker-build-serve: ## Build the serving (inference) image (FROM roshbeed/ai-residency-base)
	docker build -f Dockerfile.serve -t roshbeed/ai-residency-week1-content-scoring-serve:latest .

docker-push-train: ## Push the training image (for a RunPod/AWS GPU host to pull)
	docker push roshbeed/ai-residency-week1-content-scoring-train:latest

docker-push-serve: ## Push the serving image (for Kubernetes to pull)
	docker push roshbeed/ai-residency-week1-content-scoring-serve:latest

docker-up: ## Start the serve service
	docker compose up --build serve

docker-train: ## Run the train container once, profile-gated (never runs on a bare `up`)
	docker compose --profile train run --rm train

docker-down: ## Stop and remove all containers started by docker compose
	docker compose down

clean: ## Remove local venvs/caches/offline wandb runs (NOT data/models/artifacts)
	rm -rf .venv .venv-run .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -not -path "./data/*" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf wandb

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
