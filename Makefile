.DEFAULT_GOAL := help
.PHONY: install fmt lint typecheck test check \
        base base-train build push train serve clean help

# Every member of the workspace. Derived from the directory listing, so a new
# service is picked up the moment its directory exists.
SERVICES := $(notdir $(patsubst %/,%,$(dir $(wildcard services/*/pyproject.toml))))
# Every service has a test suite, the frontend included -- it used to be excluded here
# because it had none, which is what let it drift furthest from the rest of the repo.
CHECKABLE := $(SERVICES)

# Narrow any target to one member: `make test SERVICE=word-embeddings`.
# Unset means all of them.
SERVICE ?=
TARGETS := $(if $(SERVICE),$(SERVICE),$(SERVICES))

# Libraries are checked alongside services, and derived rather than listed. The check
# loops used to name libs/common and nothing else, so libs/vision-transformer was never
# type-checked on its own and a new library would have been silently skipped.
#
# Defined after TARGETS on purpose: `:=` expands immediately, so referring to TARGETS
# above this line silently yields an empty list and every check loop does nothing.
LIBS := $(patsubst %/,%,$(dir $(wildcard libs/*/pyproject.toml)))
MEMBERS := $(LIBS) $(addprefix services/,$(TARGETS))

REGISTRY ?= roshb
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
	@for m in $(MEMBERS); do \
		echo "--- mypy $$m"; \
		(cd $$m && uv run --project . mypy --config-file ../../pyproject.toml .) || exit 1; \
	done

# pytest exits 5 on "no tests collected", which is a fact about a member rather than a
# failure -- libs/vision-transformer has no suite of its own, being covered through the
# two ViT services that use it.
test: ## Run pytest, per member
	@for m in $(MEMBERS); do \
		if [ -d $$m/tests ]; then \
			echo "--- pytest $$m"; \
			(cd $$m && uv run --project . pytest -q) || exit 1; \
		else \
			echo "--- $$m has no tests/"; \
		fi; \
	done

check: lint typecheck test ## Everything CI runs

## --- Images --------------------------------------------------------------
## The build context is the repository root, not the service directory: an image
## needs the root pyproject.toml, the shared uv.lock and libs/common, none of
## which sit inside a service. Both Dockerfiles are still per service.

# Training images target the GPU host's architecture, not this machine's. Built
# on an Apple Silicon Mac without this they come out linux/arm64 and will not run
# on an x86-64 GPU instance -- and an emulated amd64 build of a ~10GB CUDA image
# is slow enough that CI or the GPU box itself is the right place for it.
TRAIN_PLATFORM ?= linux/amd64

# Each base is tagged twice: once under the registry name (what gets pushed) and once
# under `local/` (what the service builds actually consume). The second tag is not
# cosmetic. buildx resolves any registry-qualified `FROM` against the registry, so a
# freshly built roshb/ai-residency-base-serve:latest is ignored in favour of the
# published one -- silently, since the published image is perfectly valid. `local/...`
# exists nowhere upstream, so there is nothing to resolve to but the image on this
# machine. Same trap as the kind node's separate image store, one layer up.
LOCAL_BASE       := local/ai-residency-base-serve:$(IMAGE_TAG)
LOCAL_BASE_TRAIN := local/ai-residency-base-train:$(IMAGE_TAG)

base: ## Build the CPU base image every serving image builds FROM
	docker build -f Dockerfile.base \
		-t $(REGISTRY)/ai-residency-base-serve:$(IMAGE_TAG) -t $(LOCAL_BASE) .

base-train: ## Build the CUDA base image every training image builds FROM
	docker build --platform $(TRAIN_PLATFORM) -f Dockerfile.base \
		--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
		--build-arg TORCH_LOCAL=+cu126 \
		-t $(REGISTRY)/ai-residency-base-train:$(IMAGE_TAG) -t $(LOCAL_BASE_TRAIN) .

# Default to the local-only tags, so `make base && make build SERVICE=x` builds against
# the base you just built rather than the published one. CI overrides both with the
# registry name at that commit's sha- tag, which is what makes a published image
# reproducible from a single identifier.
#
#   make build SERVICE=x BASE_IMAGE=roshb/ai-residency-base-serve:sha-abc123
BASE_IMAGE ?= $(LOCAL_BASE)
BASE_TRAIN_IMAGE ?= $(LOCAL_BASE_TRAIN)

build: ## Build a service's images: make build SERVICE=word-embeddings
	@test -n "$(SERVICE)" || (echo "Usage: make build SERVICE=<name>   ($(SERVICES))" && exit 1)
	docker build -f services/$(SERVICE)/Dockerfile.serve \
		$(if $(filter frontend,$(SERVICE)),,--build-arg BASE_IMAGE=$(BASE_IMAGE)) \
		-t $(REGISTRY)/ai-residency-$(SERVICE)$(if $(filter frontend,$(SERVICE)),,-serve):$(IMAGE_TAG) .
	@# `if`, not `test -f ... && docker build ... || true`. Shell precedence applies
	@# that trailing `|| true` to the whole && list, so a training image that failed
	@# to build exited 0 and this target reported success -- which is exactly how a
	@# broken build went unnoticed until someone tried to run the image.
	@#
	@# A *missing* CUDA base is skipped rather than failed, though: it is a ~10GB
	@# emulated build that a laptop rarely wants, and CI and the GPU host are where
	@# training images are actually built. A base that IS present and a build that
	@# then breaks still fails, which is the case worth catching.
	@if [ ! -f services/$(SERVICE)/Dockerfile.train ]; then \
		echo "services/$(SERVICE) has no Dockerfile.train; skipping the training image"; \
	elif ! docker image inspect $(BASE_TRAIN_IMAGE) >/dev/null 2>&1; then \
		echo "$(BASE_TRAIN_IMAGE) is not built; skipping the training image (run 'make base-train' first)"; \
	else \
		docker build --platform $(TRAIN_PLATFORM) -f services/$(SERVICE)/Dockerfile.train \
			--build-arg BASE_IMAGE=$(BASE_TRAIN_IMAGE) \
			-t $(REGISTRY)/ai-residency-$(SERVICE)-train:$(IMAGE_TAG) . ; \
	fi

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
