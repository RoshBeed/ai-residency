        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

train:  ## Train the CLIP-to-Qwen3 adapter on Flickr30k
	uv run python -m multimodal_captioning.training

serve:  ## Run the FastAPI backend on :8000
	uv run python -m serving.api

zero-shot:  ## Run Part 1: CLIP zero-shot classification over data/images/ (exits when done)
	uv run python -m multimodal_captioning.zero_shot

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
