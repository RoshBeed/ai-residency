.DEFAULT_GOAL := help
.PHONY: install fmt lint typecheck test check \
        build push train serve new-service sync-dockerfiles clean help

# Every member of the workspace. Derived, so a new service is picked up the moment
# `make new-service` creates it -- there is no list to remember to update.
SERVICES := $(notdir $(patsubst %/,%,$(dir $(wildcard services/*/pyproject.toml))))
CHECKABLE := $(filter-out frontend,$(SERVICES))

# Narrow any target to one member: `make test SERVICE=word-embeddings`.
# Unset means all of them.
SERVICE ?=
TARGETS := $(if $(SERVICE),$(SERVICE),$(SERVICES))

REGISTRY ?= roshbeed
IMAGE_TAG ?= latest

install: ## Create the workspace venv and install every member's dependencies
	uv sync --all-packages

## --- Quality -------------------------------------------------------------
## One ruff and one mypy configuration, in the root pyproject.toml, applied to
## every member. mypy runs per service rather than once over the tree: each
## service has its own top-level `config` package, so a single invocation sees
## nine modules called `config` and refuses to start. Per service is also how
## the containers actually run -- one service on the path, not nine.

fmt: ## Format and auto-fix the whole workspace
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Check formatting and lint rules without changing anything
	uv run ruff format --check .
	uv run ruff check .

typecheck: ## Run mypy in strict mode, per member
	@cd libs/common && uv run --project . mypy --config-file ../../pyproject.toml .
	@for s in $(TARGETS); do \
		echo "--- mypy services/$$s"; \
		(cd services/$$s && uv run --project . mypy --config-file ../../pyproject.toml .) || exit 1; \
	done

test: ## Run pytest, per member
	@cd libs/common && uv run --project . pytest -q
	@for s in $(filter $(CHECKABLE),$(TARGETS)); do \
		echo "--- pytest services/$$s"; \
		(cd services/$$s && uv run --project . pytest -q) || exit 1; \
	done

check: lint typecheck test ## Everything CI runs

## --- Images --------------------------------------------------------------
## The build context is the repository root, not the service directory: an image
## needs the root pyproject.toml, the shared uv.lock and libs/common, none of
## which sit inside a service. Both Dockerfiles are still per service.

build: ## Build a service's images: make build SERVICE=word-embeddings
	@test -n "$(SERVICE)" || (echo "Usage: make build SERVICE=<name>   ($(SERVICES))" && exit 1)
	docker build -f services/$(SERVICE)/Dockerfile.serve \
		-t $(REGISTRY)/ai-residency-$(SERVICE)$(if $(filter frontend,$(SERVICE)),,-serve):$(IMAGE_TAG) .
	@test -f services/$(SERVICE)/Dockerfile.train && \
		docker build -f services/$(SERVICE)/Dockerfile.train \
			-t $(REGISTRY)/ai-residency-$(SERVICE)-train:$(IMAGE_TAG) . || true

push: ## Push a service's images: make push SERVICE=word-embeddings
	@test -n "$(SERVICE)" || (echo "Usage: make push SERVICE=<name>" && exit 1)
	docker push $(REGISTRY)/ai-residency-$(SERVICE)$(if $(filter frontend,$(SERVICE)),,-serve):$(IMAGE_TAG)
	-docker push $(REGISTRY)/ai-residency-$(SERVICE)-train:$(IMAGE_TAG)

## --- Per-service pipelines -----------------------------------------------
## Training and serving differ per service -- word-embeddings has a corpus stage
## and a sweep, vector-search-retrieval has three training stages and a Redis
## index, vision-transformers has two model families. Those targets stay in each
## service's own Makefile; these two just forward to them.

train: ## Train one service: make train SERVICE=word-embeddings
	@test -n "$(SERVICE)" || (echo "Usage: make train SERVICE=<name>" && exit 1)
	$(MAKE) -C services/$(SERVICE) train

serve: ## Serve one service locally: make serve SERVICE=word-embeddings
	@test -n "$(SERVICE)" || (echo "Usage: make serve SERVICE=<name>" && exit 1)
	$(MAKE) -C services/$(SERVICE) serve

sync-dockerfiles: ## Regenerate the workspace-manifest COPY block in every Dockerfile
	@python3 scripts/sync_dockerfile_manifests.py

new-service: ## Scaffold a new service from the template: make new-service NAME=my-thing
	@test -n "$(NAME)" || (echo "Usage: make new-service NAME=<name>" && exit 1)
	@test ! -d services/$(NAME) || (echo "services/$(NAME) already exists" && exit 1)
	cp -R services/_template services/$(NAME)
	@python3 scripts/rename_service.py $(NAME)
	uv lock
	@echo "services/$(NAME) is in the workspace, linted, and ready for CI."

clean: ## Remove caches and the workspace venv (never data/ or checkpoints/)
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Services: $(SERVICES)"
	@echo "Service-specific targets (train stages, sweeps, index builds) live in"
	@echo "each service's own Makefile: make -C services/<name> help"
