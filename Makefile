.DEFAULT_GOAL := help
.PHONY: install fmt lint typecheck test check \
        base base-train build build-serve run-serve stop-serve local push train serve clean help \
        up down status local-up local-cluster

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
	@# deploy/ is not a workspace member, so the loop below never reaches it -- and
	@# `deploy/images.yaml` is what the cluster actually deploys, which makes it worth
	@# more coverage than most packages here, not less.
	@echo "--- pytest deploy"
	@uv run pytest -q deploy/
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

# Deploy targets live in deploy/Makefile. These pass through so `make local-up` works
# from the repository root as well -- `make local` is here and `make local-up` is there,
# which is a distinction nobody should have to know, and the error you get otherwise
# ("No rule to make target") says nothing about the real problem.
up down status local-up local-cluster: ## Deploy targets: see `make -C deploy help`
	@# `origin`, not a bare `$(if $(SERVICES),...)`: this Makefile already defines
	@# SERVICES as every service in the workspace, so a plain check is always true and
	@# would forward the whole list -- _template included -- to a target expecting the
	@# handful the caller asked for.
	@$(MAKE) -C deploy $@ $(if $(filter command line,$(origin SERVICES)),SERVICES="$(SERVICES)")

local: ## Build one service natively for this machine: make local SERVICE=word-embeddings
	@test -n "$(SERVICE)" || (echo "Usage: make local SERVICE=<name>   ($(SERVICES))" && exit 1)
	@# Published images are linux/amd64: the cluster, the GPU host and CI runners are
	@# all x86-64, and building arm64 for them meant QEMU emulation costing 47% of CI
	@# build time for an architecture nothing in the deployment path runs. On Apple
	@# Silicon the published image therefore runs emulated. This builds it natively
	@# instead -- base first, because a service image is only as native as what it
	@# builds FROM.
	$(MAKE) base
	$(MAKE) build SERVICE=$(SERVICE)
	@echo ""
	@echo "Built natively for $$(docker version --format '{{.Server.Arch}}')."
	@echo "Run it:  docker run --rm -p 8000:8000 $(REGISTRY)/$(IMAGE_STEM):$(IMAGE_TAG)"

# The training entrypoint, read from the service's own pyproject.toml. One root
# Dockerfile.train builds every training image, and this is the only value that
# differs between them -- kept beside the code it names rather than in a table here.
train_module = $(shell python3 -c "import tomllib;print(tomllib.load(open('services/$(1)/pyproject.toml','rb'))['tool']['ai-residency']['train-module'])")

# One root Dockerfile per image kind, selected by --build-arg, rather than one per
# service. The frontend is the exception in both directions: it has its own
# Dockerfile.serve (no torch base, no FastAPI app) and no training image at all.
SERVE_DOCKERFILE = $(if $(filter frontend,$(SERVICE)),services/frontend/Dockerfile.serve,Dockerfile.serve)

# A Docker repository name matches [a-z0-9]+([._-][a-z0-9]+)*, so `ai-residency-_template`
# is not one -- two separators in a row. `make build SERVICE=_template` had always failed
# on "invalid reference format", which nothing noticed because CI never built the
# template. It is in the check matrix now, and a template you cannot build is not the
# "working one, not a skeleton" it is documented as, so the leading underscore is
# dropped from the image name. Nothing publishes this image; it exists to prove the
# path works for a service that does not exist yet.
IMAGE_STEM = ai-residency-$(patsubst _%,%,$(SERVICE))$(if $(filter frontend,$(SERVICE)),,-serve)

build: ## Build a service's images: make build SERVICE=word-embeddings
	@test -n "$(SERVICE)" || (echo "Usage: make build SERVICE=<name>   ($(SERVICES))" && exit 1)
	docker build -f $(SERVE_DOCKERFILE) \
		--build-arg SERVICE=$(SERVICE) \
		$(if $(filter frontend,$(SERVICE)),,--build-arg BASE_IMAGE=$(BASE_IMAGE)) \
		-t $(REGISTRY)/$(IMAGE_STEM):$(IMAGE_TAG) .
	@# `if`, not `test -f ... && docker build ... || true`. Shell precedence applies
	@# that trailing `|| true` to the whole && list, so a training image that failed
	@# to build exited 0 and this target reported success -- which is exactly how a
	@# broken build went unnoticed until someone tried to run the image.
	@#
	@# A *missing* CUDA base is skipped rather than failed, though: it is a ~10GB
	@# emulated build that a laptop rarely wants, and CI and the GPU host are where
	@# training images are actually built. A base that IS present and a build that
	@# then breaks still fails, which is the case worth catching.
	@if [ "$(SERVICE)" = "frontend" ]; then \
		echo "services/frontend has no training image; skipping"; \
	elif ! docker image inspect $(BASE_TRAIN_IMAGE) >/dev/null 2>&1; then \
		echo "$(BASE_TRAIN_IMAGE) is not built; skipping the training image (run 'make base-train' first)"; \
	else \
		docker build --platform $(TRAIN_PLATFORM) -f Dockerfile.train \
			--build-arg SERVICE=$(SERVICE) \
			--build-arg TRAIN_MODULE=$(call train_module,$(SERVICE)) \
			--build-arg BASE_IMAGE=$(BASE_TRAIN_IMAGE) \
			-t $(REGISTRY)/ai-residency-$(patsubst _%,%,$(SERVICE))-train:$(IMAGE_TAG) . ; \
	fi

# Everything needed to run one service and its UI, and nothing else. Three things this
# gets right that the individual targets leave to the caller:
#
#   * `base` first. Without it `build` tries to *pull* local/ai-residency-base-serve,
#     which does not exist in any registry, and fails with "pull access denied" -- an
#     auth error that is really a missing local image.
#   * `REGISTRY=local`. The default is the published name, so a plain `make build`
#     silently shadows whatever was pulled from Docker Hub and you can no longer tell
#     which image is running.
#   * the frontend, which is a separate service and easy to forget until the page says
#     the backend is not ready.
#
# The training image is skipped: it is a ~9GB CUDA build that serving never needs.
build-serve: ## Build one service + the UI, ready to run: make build-serve SERVICE=content-scoring
	@test -n "$(SERVICE)" || (echo "Usage: make build-serve SERVICE=<name>   ($(SERVICES))" && exit 1)
	@test "$(SERVICE)" != "frontend" || (echo "build-serve takes a backend; the UI is built for you" && exit 1)
	$(MAKE) base REGISTRY=local
	$(MAKE) build SERVICE=$(SERVICE) REGISTRY=local
	$(MAKE) build SERVICE=frontend REGISTRY=local
	@echo ""
	@echo "Built. Run them with:"
	@echo ""
	@echo "  make run-serve SERVICE=$(SERVICE)"

run-serve: ## Run a locally built service + UI: make run-serve SERVICE=content-scoring
	@test -n "$(SERVICE)" || (echo "Usage: make run-serve SERVICE=<name>   ($(SERVICES))" && exit 1)
	@# Checked before `docker run`, which would otherwise try to *pull* an image that
	@# exists in no registry and fail with "pull access denied" -- an auth error that is
	@# really "you have not built this yet". Same misleading failure as a missing base.
	@docker image inspect local/$(IMAGE_STEM) >/dev/null 2>&1 \
		|| (echo "local/$(IMAGE_STEM) is not built. Run: make build-serve SERVICE=$(SERVICE)" && exit 1)
	@docker image inspect local/ai-residency-frontend >/dev/null 2>&1 \
		|| (echo "local/ai-residency-frontend is not built. Run: make build-serve SERVICE=$(SERVICE)" && exit 1)
	@# Removed first so the target is safe to run twice; `docker run` errors on a name
	@# that is already taken, which is the most common way this goes wrong.
	-@docker rm -f $(SERVICE) frontend >/dev/null 2>&1
	-@docker network create ai-residency-local >/dev/null 2>&1
	docker run --rm -d --name $(SERVICE) --network ai-residency-local -p 8000:8000 \
		local/$(IMAGE_STEM)
	@# The container name, not localhost: inside the frontend container localhost is the
	@# frontend, so the page would report "backend isn't ready" and look exactly like a
	@# backend that is down.
	docker run --rm -d --name frontend --network ai-residency-local -p 8501:8501 \
		-e AI_RESIDENCY_PAGES=$(SERVICE) \
		-e $(shell echo $(SERVICE) | tr 'a-z-' 'A-Z_')_API_URL=http://$(SERVICE):8000 \
		local/ai-residency-frontend
	@echo "waiting for $(SERVICE) to pull its checkpoints..."
	@curl -fsS --retry 90 --retry-delay 2 --retry-all-errors http://localhost:8000/ready >/dev/null \
		&& echo "" && echo "  UI:      http://localhost:8501" && echo "  Backend: http://localhost:8000/docs"

stop-serve: ## Stop what run-serve started
	-@docker rm -f $(SERVICE) frontend >/dev/null 2>&1
	-@docker network rm ai-residency-local >/dev/null 2>&1
	@echo "stopped"

push: ## Push a service's images: make push SERVICE=word-embeddings
	@test -n "$(SERVICE)" || (echo "Usage: make push SERVICE=<name>" && exit 1)
	docker push $(REGISTRY)/$(IMAGE_STEM):$(IMAGE_TAG)
	-docker push $(REGISTRY)/ai-residency-$(patsubst _%,%,$(SERVICE))-train:$(IMAGE_TAG)

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
