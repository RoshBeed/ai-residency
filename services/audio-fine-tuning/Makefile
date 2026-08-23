.PHONY: install train demo serve prepare-data download-data test fmt lint typecheck check \
        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

install:  ## Install the package and dev dependencies
	uv sync

train:  ## Run the Part 1 fine-tuning loop and write a checkpoint
	uv run python -m audio_fine_tuning.training

demo:  ## Print the base-vs-fine-tuned transcript comparison for data/audio.mp3
	uv run python -m audio_fine_tuning.demo

serve:  ## Run the FastAPI backend on :8000
	uv run python -m serving.api

prepare-data:  ## Part 2 pipeline: prepare -> manifest -> tokens -> explore
	uv run python -m data_prep.prepare_dataset
	uv run python -m data_prep.create_manifest
	uv run python -m audio_fine_tuning.add_emotion_tokens
	uv run python -m data_prep.explore_data

download-data:  ## Fetch the raw corpora (EmoDB downloads; CREMA-D is a slow git-lfs clone)
	uv run python -m data_prep.download_dataset

test:  ## Run the test suite
	uv run pytest -q

fmt:  ## Format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Check formatting and lint rules without changing anything
	uv run ruff format --check .
	uv run ruff check .

typecheck:  ## Run mypy in strict mode
	uv run mypy config audio_fine_tuning serving data_prep tests

check: lint typecheck test  ## Everything CI would run

# Note (Apple Silicon): Docker Desktop's Linux VM has no MPS passthrough, so every
# docker-* target runs on CPU regardless of host hardware. Use the native targets above
# for real M-series acceleration — roughly 3-4x faster at this repo's scale.

docker-build-train:  ## Build the training image, FROM roshbeed/ai-residency-base
	docker build -f Dockerfile.train -t roshbeed/ai-residency-week5-train:latest .

docker-build-serve:  ## Build the serving (inference) image, FROM roshbeed/ai-residency-base
	docker build -f Dockerfile.serve -t roshbeed/ai-residency-week5-serve:latest .

docker-push-train:  ## Push the training image (for a RunPod/AWS GPU host to pull)
	docker push roshbeed/ai-residency-week5-train:latest

docker-push-serve:  ## Push the serving image (for Kubernetes to pull)
	docker push roshbeed/ai-residency-week5-serve:latest

docker-up:  ## Start the serve service
	docker compose up --build serve

docker-train:  ## Run training in a container (profile-gated)
	docker compose --profile train run --rm train

docker-down:  ## Stop everything and remove volumes
	docker compose down -v

clean:  ## Remove caches and build artifacts (never touches datasets/ or checkpoints/)
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info wandb
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
