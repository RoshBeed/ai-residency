.PHONY: install run docker-build docker-push help
.DEFAULT_GOAL := help

install: ## Install the frontend's own lightweight deps (streamlit + requests only)
	uv sync

run: ## Run the app locally against whatever backends are already up
	uv run streamlit run app.py

docker-build: ## Build the published frontend image
	docker build -t roshbeed/ai-residency-frontend:latest .

docker-push: ## Push the image to Docker Hub (needs `docker login` first)
	docker push roshbeed/ai-residency-frontend:latest

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
