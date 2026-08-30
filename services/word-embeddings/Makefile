.DEFAULT_GOAL := help
.PHONY: train train-sample time sweep visualise evaluate serve test help

# The corpus is text8, fetched from the Hub on first use, so there is no corpus stage
# to run first. There used to be a `train-corpus` target here building a Hacker News
# extract for content-scoring; that service owns its own ETL now.
train: ## Train word vectors on text8 (~75s/epoch on an M1 Max; 10 epochs by default)
	uv run python -m word_embeddings.train

train-sample: ## Train on a tiny slice of the corpus — seconds, writes a separate checkpoint
	uv run python -m word_embeddings.train --sample

time: ## Measure throughput and project how long a full run takes, without training
	uv run python -m word_embeddings.train --time-only

# HOURS bounds the sweep by wall clock, COUNT by number of runs; HOURS is usually
# what you want. Run times here span 6 to 31 minutes depending on architecture,
# embedding size and how much the subsampling threshold keeps, so a COUNT chosen for
# an average finishes anywhere from half to double the intended budget. Stopping an
# agent early costs nothing: every finished run is already recorded, and Bayes reads
# them back on the next agent.
#
# caffeinate because an overnight sweep on a laptop otherwise stops when it sleeps.
# timeout is GNU coreutils, not macOS base -- `brew install coreutils`.
sweep: ## Run a W&B sweep agent: make sweep SWEEP=<id> [HOURS=12] [COUNT=n]
	@test -n "$(SWEEP)" || (echo "Usage: make sweep SWEEP=<sweep id> [HOURS=12] [COUNT=n]   (create one with: wandb sweep sweep.yaml)" && exit 1)
	caffeinate -is $(if $(HOURS),timeout $(HOURS)h,) uv run wandb agent $(if $(COUNT),--count $(COUNT),) $(SWEEP)

evaluate: ## Score the trained checkpoint on WordSim-353, SimLex-999 and the Google analogies
	uv run python -m word_embeddings.evaluate

visualise: ## Score the checkpoint and write a t-SNE plot of the embedding space to assets/
	uv run python -m word_embeddings.evaluate --visualise

serve: ## Run the FastAPI nearest-neighbour/analogy service locally (no Docker)
	uv run python -m serving.api

test: ## Run the test suite
	uv run pytest -q

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
