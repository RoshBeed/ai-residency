        docker-build-base docker-push-base docker-build-train docker-build-serve \
        docker-push-train docker-push-serve docker-up docker-train docker-down clean help

# ai-residency-common is a private repo, so only building the shared base image needs a
# token — `gh auth token` supplies it without it ever being stored in a file.
# Every image below builds FROM that base and never touches git.
GITHUB_TOKEN ?= $(shell gh auth token)
export GITHUB_TOKEN

train:  ## Run training
	uv run python -m template.training

serve:  ## Run the FastAPI backend on :8000
	uv run python -m serving.api

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
