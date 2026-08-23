.DEFAULT_GOAL := help
.PHONY: train predict serve test help

train: ## Train the regression head over the frozen CBOW embeddings
	uv run python -m content_scoring.train

predict: ## Score a title from the command line: make predict TITLE="..."
	@test -n "$(TITLE)" || (echo "Usage: make predict TITLE='your title here'" && exit 1)
	uv run python -m content_scoring.predict "$(TITLE)"

serve: ## Run the FastAPI prediction service locally (no Docker)
	uv run python -m serving.api

test: ## Run the test suite
	uv run pytest -q

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
