        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

# ai-residency-common is a private repo, so only building the shared base image needs a
# token — `gh auth token` supplies it without it ever being stored in a file.
# Every image below builds FROM that base and never touches git.
GITHUB_TOKEN ?= $(shell gh auth token)
export GITHUB_TOKEN

train: train-classifier  ## Alias for the faster of the two training runs

train-classifier:  ## Train the single-digit classifier
	uv run python -m models.classifier.trainer

train-seq2seq:  ## Train the multi-digit seq2seq model
	uv run python -m models.seq2seq.trainer

serve:  ## Run the FastAPI backend on :8000 (loads both committed checkpoints)
	uv run python -m serving.api

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
