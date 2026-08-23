.PHONY: install fmt lint typecheck test check clean help

install:  ## Install the package and dev dependencies
	uv sync

fmt:  ## Format the code
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Check formatting and lint rules without changing anything
	uv run ruff format --check .
	uv run ruff check .

typecheck:  ## Run mypy in strict mode
	uv run mypy src tests

test:  ## Run the test suite
	uv run pytest -q

check: lint typecheck test  ## Everything CI would run

clean:  ## Remove caches and build artifacts
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
