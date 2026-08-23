.DEFAULT_GOAL := help
.PHONY: train serve test help

train:  ## Run training
	uv run python -m template.training

serve:  ## Run the FastAPI backend on :8000
	uv run python -m serving.api

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
