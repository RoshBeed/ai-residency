.PHONY: install train serve test fmt lint typecheck check \
        docker-build-base docker-push-base docker-build-train docker-build-serve \
        docker-push-train docker-push-serve docker-up docker-train docker-down clean help

# ai-residency-common is a private repo, so only building the shared base image needs a
# token — `gh auth token` supplies it without it ever being stored in a file.
# Every image below builds FROM that base and never touches git.
GITHUB_TOKEN ?= $(shell gh auth token)
export GITHUB_TOKEN

install:  ## Install the package and dev dependencies
	uv sync

train:  ## Run training
	uv run python -m template.training

serve:  ## Run the FastAPI backend on :8000
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
	uv run mypy config template serving tests

check: lint typecheck test  ## Everything CI would run

docker-build-base:  ## Build the shared roshbeed/ai-residency-base image (this repo is its source)
	docker build -f Dockerfile.base --secret id=gh_token,env=GITHUB_TOKEN -t roshbeed/ai-residency-base:local .

docker-push-base:  ## Push roshbeed/ai-residency-base to Docker Hub (needs `docker login` first)
	docker tag roshbeed/ai-residency-base:local roshbeed/ai-residency-base:latest
	docker push roshbeed/ai-residency-base:latest

docker-build-train:  ## Build the training image
	docker build -f Dockerfile.train -t ai-residency-template-train .

docker-build-serve:  ## Build the serving (inference) image
	docker build -f Dockerfile.serve -t ai-residency-template-serve .

docker-push-train:  ## Push the training image (for a RunPod/AWS GPU host to pull)
	docker tag ai-residency-template-train roshbeed/ai-residency-template-train:latest
	docker push roshbeed/ai-residency-template-train:latest

docker-push-serve:  ## Push the serving image (for Kubernetes to pull)
	docker tag ai-residency-template-serve roshbeed/ai-residency-template-serve:latest
	docker push roshbeed/ai-residency-template-serve:latest

docker-up:  ## Start the serve service
	docker compose up --build serve

docker-train:  ## Run training in a container (profile-gated)
	docker compose --profile train run --rm train

docker-down:  ## Stop everything and remove volumes
	docker compose down -v

clean:  ## Remove caches and build artifacts
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
