        test fmt lint typecheck check redis \
        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

# Note on train-* vs docker-train: the local targets pick up Apple Silicon's MPS backend
# where it helps, which for this repo means the CBOW stage only (~3.7x faster than CPU on
# its own batch/vocab shape). The tower training in training.py runs one triple at a time
# and benchmarked slower on MPS, so it stays on CPU. `docker-train` is always CPU-only —
# Docker Desktop's Linux VM has no GPU or MPS passthrough — so it is for portability and a
# future real GPU host, not local speed.

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
	docker run -d --name ai-residency-redis -p 6390:6379 redis/redis-stack-server:latest \
		redis-stack-server --save "" --appendonly no

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
