.PHONY: install train train-cbow train-triples train-encoder index query serve \
        test fmt lint typecheck check redis \
        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

# Note on train-* vs docker-train: the local targets pick up Apple Silicon's MPS backend
# where it helps, which for this repo means the CBOW stage only (~3.7x faster than CPU on
# its own batch/vocab shape). The tower training in training.py runs one triple at a time
# and benchmarked slower on MPS, so it stays on CPU. `docker-train` is always CPU-only —
# Docker Desktop's Linux VM has no GPU or MPS passthrough — so it is for portability and a
# future real GPU host, not local speed.

install:  ## Install the package and dev dependencies
	uv sync

train-cbow:  ## Build the vocabulary, then train CBOW word embeddings
	uv run python -m cbow.train_tokenizer
	uv run python -m cbow.train_word2vec

train-triples:  ## Build and tokenize MS MARCO triples
	uv run python -m data_prep.generate_triples
	uv run python -m data_prep.tokenize_triples

train-encoder:  ## Train the GRU dual-encoder towers
	uv run python -m models.training

train: train-cbow train-triples train-encoder  ## Run the full offline pipeline in order

index:  ## Embed the corpus with the trained doc tower and index it in Redis
	uv run python -m retrieval.save_doc_embeddings_to_redis

query:  ## Index the corpus, then the interactive query CLI (needs Redis running)
	uv run python -m retrieval.save_doc_embeddings_to_redis
	uv run python -m retrieval.query_redis_ann

serve:  ## Run the FastAPI backend on :8000 (needs Redis running and a populated index)
	uv run python -m serving.api

redis:  ## Start a throwaway RediSearch-capable Redis on :6390
	docker run -d --name ai-residency-week2-redis -p 6390:6379 redis/redis-stack-server:latest \
		redis-stack-server --save "" --appendonly no

test:  ## Run the test suite
	uv run pytest -q

fmt:  ## Format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Check formatting and lint rules without changing anything
	uv run ruff format --check .
	uv run ruff check .

typecheck:  ## Run mypy in strict mode
	uv run mypy config cbow data_prep models retrieval serving tests

check: lint typecheck test  ## Everything CI would run

docker-build-train:  ## Build the training image
	docker build -f Dockerfile.train -t roshbeed/ai-residency-week2-train:latest .

docker-build-serve:  ## Build the serving (inference) image
	docker build -f Dockerfile.serve -t roshbeed/ai-residency-week2-serve:latest .

docker-push-train:  ## Push the training image (for a RunPod/AWS GPU host to pull)
	docker push roshbeed/ai-residency-week2-train:latest

docker-push-serve:  ## Push the serving image (for Kubernetes to pull)
	docker push roshbeed/ai-residency-week2-serve:latest

docker-up:  ## Start Redis + the serve service
	docker compose up --build serve

docker-train:  ## Run the full training pipeline in the train container
	docker compose --profile train run --rm train

docker-down:  ## Stop and remove all Compose services
	docker compose down

clean:  ## Remove venvs, caches and wandb runs (never checkpoints or generated data)
	rm -rf .venv .venv-run .pytest_cache .mypy_cache .ruff_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	rm -rf wandb

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
