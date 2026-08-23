.DEFAULT_GOAL := help
.PHONY: run help

run: ## Run the app locally against whatever backends are already up
	uv run streamlit run app.py

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
