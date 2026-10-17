.DEFAULT_GOAL := help
.PHONY: evaluate train serve zero-shot test help

train:  ## Train the CLIP-to-Qwen3 adapter on Flickr30k
	uv run python -m multimodal_captioning.training

evaluate:  ## Score the adapter by CLIP agreement against the untrained floor and human ceiling
	uv run python -m multimodal_captioning.evaluate

serve:  ## Run the FastAPI backend on :8000
	uv run python -m serving.api

zero-shot:  ## Run Part 1: CLIP zero-shot classification over data/images/ (exits when done)
	uv run python -m multimodal_captioning.zero_shot

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
