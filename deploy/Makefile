.PHONY: pull up down logs status index-retrieval load-images pin-images local-up local-cluster \
        helm-install helm-upgrade helm-uninstall helm-status helm-train helm-train-logs \
        helm-index-retrieval help
.DEFAULT_GOAL := help

# Runtime credentials for the backends and training Jobs. Read from the portfolio's
# own .env, which sits one level up beside the repos, so nothing here or in
# values.yaml ever holds a token. Missing is fine: the stack still runs, it just
# cannot reach the Hub or log to W&B.
# The repository root first, then one level above it -- the portfolio's own .env
# lived beside the repos before they were one repo.
ENV_FILE ?= $(firstword $(wildcard ../.env ../../.env))
HF_TOKEN ?= $(shell [ -f $(ENV_FILE) ] && grep -E '^HF_TOKEN=' $(ENV_FILE) | cut -d= -f2-)
WANDB_API_KEY ?= $(shell [ -f $(ENV_FILE) ] && grep -E '^WANDB_API_KEY=' $(ENV_FILE) | cut -d= -f2-)
export HF_TOKEN
export WANDB_API_KEY

# Every compose invocation, in one place. `--env-file images.env` is what pins the
# images: docker-compose.yml carries `${<SERVICE>_REF:-:latest}` and that file supplies
# the `@sha256:...` half. Without it -- `make local-up`, which passes /dev/null -- every
# service resolves to `:latest`, which is what a locally-built image is tagged.
#
# Not named `.env`: that is gitignored repo-wide because it holds credentials, and a
# committed generated file wearing the same name is a trap for whoever reads this next.
COMPOSE := docker compose --env-file compose/images.env -f compose/docker-compose.yml
COMPOSE_LOCAL := docker compose --env-file /dev/null -f compose/docker-compose.yml

HELM_RELEASE := ai-residency
HELM_NAMESPACE := ai-residency
HELM_CHART := ./helm/ai-residency
# Passed via a temp file rather than on the command line: an argument to helm is
# visible in `ps` to every user on the machine for as long as the install runs.
# --set-file reads the value from a path instead, and the file is created under
# `umask 077` and deleted straight after.
HELM_SECRETS = $(shell umask 077; \
	d=$$(mktemp -d); \
	printf '%s' '$(HF_TOKEN)' > $$d/hf; \
	printf '%s' '$(WANDB_API_KEY)' > $$d/wandb; \
	echo "--set-file secrets.hfToken=$$d/hf --set-file secrets.wandbApiKey=$$d/wandb")
# Suffixes each training Job's name. A Job's pod template is immutable, so a second
# run needs a new name; the epoch second is the cheapest thing that is always unique.
TRAIN_RUN ?= $(shell date +%s)

# Every image is pulled from Docker Hub (roshb/ai-residency-<service>-serve / -train),
# built and pushed from the repository root with `make build SERVICE=<name>` and
# `make push SERVICE=<name>`. This directory only deploys them.

pull: ## Pull every backend + frontend image from Docker Hub
	$(COMPOSE) pull

# Derived from the compose file rather than written out, so a renamed or added service
# cannot leave this printing names that no longer exist -- which is what it used to do,
# advertising "vision-transformers:8003" and "audio-fine-tuning:8005" long after both
# were split and renamed.
#
# `compose config`, not the awk program that used to pattern-match service headers and
# `"80NN:8000"` lines out of the YAML. Compose resolves the file the same way it does
# when it runs it -- overrides, anchors and variable substitution included -- which a
# line-oriented match cannot. Selected on the *container* port being 8000, which is what
# makes this the backends and not the frontend (8501) or Redis.
#
# Needs jq. Empty rather than wrong if it is missing: `status` then lists no backends,
# which is visibly nothing rather than a plausible-looking stale list.
BACKENDS := $(shell docker compose -f compose/docker-compose.yml config --format json 2>/dev/null \
	| jq -r '.services | to_entries[] | select(any(.value.ports[]?; .target == 8000)) | "\(.key):\(.value.ports[0].published)"' \
	| sort)
PORTS := $(shell echo "$(BACKENDS)" | tr ' ' '\n' | cut -d: -f2 | sort)

# The frontend's compose service depends_on all nine backends, so a bare `up` starts
# the lot -- around 19GB resident once every model is loaded. SERVICES= runs the
# frontend against a chosen subset instead, which is the difference between this being
# demoable on a 16GB laptop and not.
#
# Naming services starts those plus whatever each one declares, so redis arrives with
# vector-search-retrieval on its own. The frontend used to depends_on all nine, which
# made every subset need --no-deps -- and that stripped redis back out, so it had to be
# listed here by hand. The frontend declares no backends now; see docker-compose.yml.
SUBSET := $(SERVICES)

up: ## Start everything, or a subset: make up SERVICES="word-embeddings vit-classifier"
	@if [ -n "$(SERVICES)" ]; then \
		AI_RESIDENCY_PAGES="$(SERVICES)" \
		$(COMPOSE) up -d frontend $(SUBSET); \
	else \
		$(COMPOSE) up -d; \
	fi
	@echo ""
	@echo "Frontend: http://localhost:8500"
	@if [ -n "$(SERVICES)" ]; then \
		echo "Running:  $(SUBSET)"; \
		echo "The UI shows only these pages."; \
	else \
		echo "Backends: $(BACKENDS)"; \
	fi

index-retrieval: ## Rebuild vector-search-retrieval's redis index (compose builds it on `up`; this is for re-running it)
	$(COMPOSE) exec vector-search-retrieval python -m retrieval.save_doc_embeddings_to_redis

helm-index-retrieval: ## Rebuild the cluster's vector-search-retrieval redis index (the chart's bootstrapCommand builds it on install)
	kubectl exec -n $(HELM_NAMESPACE) deploy/vector-search-retrieval -- python -m retrieval.save_doc_embeddings_to_redis

down: ## Stop and remove all containers
	$(COMPOSE) down

logs: ## Tail logs from every service
	$(COMPOSE) logs -f

status: ## Show which backends are up (/health) and which can actually serve (/ready)
	@for w in $(PORTS); do \
		health=$$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:$$w/health 2>/dev/null); \
		ready=$$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:$$w/ready 2>/dev/null); \
		[ "$$health" = "000" ] && health="down"; \
		[ "$$ready" = "000" ] && ready="down"; \
		printf 'port %s  health %s  ready %s\n' "$$w" "$$health" "$$ready"; \
	done
	@echo "ready 503 means the process is up but has no usable checkpoint -- it is out of"
	@echo "the load balancer's rotation under Kubernetes, and answers 503 here."

# Docker Desktop runs its Kubernetes node as a kind container with its own containerd
# image store. A `docker build` on the host reaches dockerd and stops there, so the
# cluster never sees it -- and because pullPolicy is IfNotPresent, it silently keeps
# running whatever image the node already had. No error, no event, nothing in
# `kubectl describe` to suggest the image is months old. Run this after every rebuild.
K8S_NODE ?= $(shell kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
IMAGE_TAG ?= latest
# Asked of Compose rather than grepped out of it. The grep this replaced matched
# `roshb/ai-residency-` out of a *comment* on line 3 of the file and fed it to
# `docker image inspect`, which then reported a nonexistent image as "not built
# locally" on every run. Resolved with no env file so every reference comes back as
# a plain `:latest` tag, which is the form this target wants anyway.
# --profile train, or the training images are missing: those services are profile-gated
# so a bare `up` never trains, and `config` honours that. Filtered to roshb/ because
# these are the images this repo builds -- Redis is pulled, never loaded.
IMAGES := $(shell $(COMPOSE_LOCAL) --profile train config --format json 2>/dev/null \
	| jq -r '[.services[].image | split(":")[0] | select(startswith("roshb/"))] | unique[]')

pin-images: ## Regenerate compose/images.env from deploy/images.yaml
	@# Only the compose env file is generated now. The chart reads images.yaml directly
	@# as a values overlay, so there is nothing to propagate into values.yaml and
	@# nothing there that can drift from the manifest. CI keeps images.yaml current;
	@# this is for regenerating after editing it by hand.
	uv run python pin_images.py
	@uv run python pin_images.py --check

local-up: ## Build, run and watch one service with the UI: make local-up SERVICES="content-scoring"
	@test -n "$(SERVICES)" || (echo 'Usage: make local-up SERVICES="word-embeddings"' && exit 1)
	@# `--env-file /dev/null`, so none of the `${<SERVICE>_REF}` variables are set and
	@# every image falls back to the `:latest` tag `make local` produces. Otherwise
	@# compose resolves the pinned digests and you build for a minute, then run
	@# someone else's image without noticing. This was a generated override file whose
	@# only job was to undo the pinning; the fallback in the image reference does it.
	$(MAKE) -C .. base
	@for s in $(SERVICES) frontend; do \
		echo "==> building $$s"; \
		$(MAKE) -C .. build SERVICE=$$s || exit 1; \
	done
	@# --force-recreate because a container left behind by a failed run gets *started*
	@# rather than recreated, and comes back with no network and no port mapping. The
	@# usual cause is port 8500 being held by a Helm release of this same chart, whose
	@# LoadBalancer Service claims it even with its pods scaled to zero --
	@# `make helm-uninstall` frees it.
	AI_RESIDENCY_PAGES="$(SERVICES)" $(COMPOSE_LOCAL) \
		up -d --force-recreate frontend $(SUBSET)
	@echo ""
	@echo "Running. UI: http://localhost:8500"
	@echo ""
	@# Watching by default, because the alternative is rebuilding a container to see a
	@# one-line change. `--no-up` because the stack is already up from the line above;
	@# watch resolves depends_on and takes no --no-deps, so it starts what it is asked
	@# for. That used to mean every published image, back when the frontend declared all
	@# nine as dependencies -- and on Apple Silicon it failed outright against the
	@# amd64-only digests of the day.
	@#
	@# WATCH=no leaves the stack up and returns, for anything that is not a dev loop.
	@if [ "$(WATCH)" = "no" ]; then \
		echo "Started detached. Stop it with: make down"; \
	else \
		echo "Watching source. Edit and the container restarts; Ctrl-C to stop watching."; \
		echo "(the stack keeps running afterwards -- 'make down' stops it)"; \
		echo ""; \
		AI_RESIDENCY_PAGES="$(SERVICES)" $(COMPOSE_LOCAL) watch --no-up $(SERVICES) frontend; \
	fi

# Every service in the workspace, derived here rather than inherited. The root
# Makefile defines a TARGETS variable and does not export it, so referring to it from
# this file expanded to nothing: `make local-cluster` with no SERVICES built no images
# at all, then installed a chart pointing at `:latest` tags the node had never been
# given. Silent, and the target's own help text advertises the no-argument form.
ALL_SERVICES := $(notdir $(patsubst %/,%,$(dir $(wildcard ../services/*/pyproject.toml))))
LOCAL_SERVICES = $(if $(SERVICES),$(SERVICES),$(filter-out _template,$(ALL_SERVICES)))

local-cluster: ## Build every image natively and run the cluster from them: make local-cluster [SERVICES="a b"]
	@# Three steps that are easy to get wrong separately: build natively, push the
	@# results into the node's *own* image store (a `docker build` on the host is
	@# invisible to it), and install *without* the images.yaml overlay so the chart asks
	@# for the tag those images carry rather than a published digest they never match.
	@#
	@# This is Skaffold's shape -- build, load, deploy, restart -- and Skaffold was
	@# considered here and turned down. Two reasons. The load step is the only part
	@# that is genuinely awkward, and it cannot be delegated: Docker Desktop runs its
	@# node as `desktop-control-plane` and does not expose it through the kind CLI, so
	@# `kind load docker-image` is unavailable and `docker save | ctr images import` is
	@# what is left either way. And published images have been multi-arch since the
	@# arm64 restore, so this target is now an escape hatch for iterating on something
	@# unpublished rather than the normal path -- a new dependency, a skaffold.yaml and
	@# a second definition of the build graph is a poor trade for that.
	$(MAKE) -C .. base
	@for s in $(LOCAL_SERVICES); do \
		echo "==> building $$s"; \
		$(MAKE) -C .. build SERVICE=$$s || exit 1; \
	done
	$(MAKE) load-images
	helm upgrade --install $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) --create-namespace \
		$(HELM_SECRETS) --set image.tag=$(IMAGE_TAG)
	@echo ""
	@echo "Cluster running locally-built images. Frontend: http://localhost:8500"

load-images: ## Import locally-built images into the cluster node's own image store
	@for img in $(IMAGES); do \
		docker image inspect $$img:$(IMAGE_TAG) >/dev/null 2>&1 || { echo "skip  $$img:$(IMAGE_TAG) (not built locally)"; continue; }; \
		printf 'load  %s:%s\n' "$$img" "$(IMAGE_TAG)"; \
		docker save $$img:$(IMAGE_TAG) | docker exec -i $(K8S_NODE) ctr -n k8s.io images import - >/dev/null; \
	done
	@echo "Roll the pods to pick them up: kubectl rollout restart deployment -n $(HELM_NAMESPACE)"

# `-f images.yaml` is what pins the release to the exact set the manifest names. It is
# a plain Helm values overlay, merged over the chart's defaults -- no propagation step,
# and nothing in values.yaml to fall out of step with it.
HELM_IMAGES := -f images.yaml

helm-install: ## Install the Helm chart onto the local Kubernetes cluster
	helm install $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) --create-namespace $(HELM_IMAGES) $(HELM_SECRETS)

helm-upgrade: ## Re-apply the Helm chart after repulling images, then roll pods to pick them up
	helm upgrade $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) $(HELM_IMAGES) $(HELM_SECRETS)
	kubectl rollout restart deployment -n $(HELM_NAMESPACE)

helm-uninstall: ## Remove the Helm release (namespace itself is left in place, not deleted)
	@# --wait, because `helm uninstall` otherwise returns before the API server has
	@# finished deleting, and an install started immediately after can have its own
	@# objects swept by the tail of the previous release -- which is how a reinstall
	@# once came up with every pod running and the frontend Service simply absent.
	@# One flag in place of the two `kubectl wait --for=delete` calls this used to run.
	@#
	@# Tolerated rather than fatal, and that is not laziness -- it was measured. On
	@# Docker Desktop the frontend's LoadBalancer can wedge in Terminating forever,
	@# because the built-in LB controller fails its own cleanup check:
	@#
	@#   SyncLoadBalancerFailed: failed to check if load balancer exists before
	@#   cleanup: container addresses should have 2 values, got 1 values
	@#
	@# Nothing in this chart can fix that, and letting it fail the target means
	@# `make helm-uninstall` errors on every teardown on the one cluster this repo is
	@# developed against. The wait still happens, which is the part that matters; what
	@# is dropped is only the exit code. If a reinstall afterwards comes up missing a
	@# Service, this is why -- clear it with:
	@#
	@#   kubectl patch svc frontend -n $(HELM_NAMESPACE) -p '{"metadata":{"finalizers":null}}' --type=merge
	helm uninstall $(HELM_RELEASE) -n $(HELM_NAMESPACE) --wait --timeout 90s || \
		echo "note: uninstall did not fully settle -- see the comment on this target"

helm-status: ## Show pod/service status for the Helm release
	kubectl get pods,svc -n $(HELM_NAMESPACE)

helm-train: ## Run a training Job on the cluster: make helm-train SERVICE=vit-classifier [EPOCHS=1]
	helm upgrade $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) $(HELM_IMAGES) $(HELM_SECRETS) \
		--set train.enabled=true \
		--set train.runName=$(TRAIN_RUN) \
		--set train.only=$(SERVICE) \
		--set train.env.AI_RESIDENCY_NUM_EPOCHS=$(or $(EPOCHS),1)
	@echo "Jobs launched. Follow with: make helm-train-logs"

helm-train-logs: ## Tail whichever training Jobs are currently running
	kubectl get jobs -n $(HELM_NAMESPACE)
	@# Selected by job-name, not by an enumerated list of service names -- the old list
	@# still named "vision-transformers-train" and "audio-fine-tuning-train", neither of
	@# which has existed since those services were split and renamed.
	kubectl logs -n $(HELM_NAMESPACE) --tail=40 --prefix --selector='job-name' 

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
