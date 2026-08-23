.DEFAULT_GOAL := help
.PHONY: train-corpus train-cbow train-cbow-sample train sweep visualise serve test help

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

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
