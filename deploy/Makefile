.PHONY: pull up down logs status index-retrieval load-images pin-images \
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
	docker compose -f compose/docker-compose.yml pull

# Derived from the compose file rather than written out, so a renamed or added service
# cannot leave this printing names that no longer exist -- which is what it used to do,
# advertising "vision-transformers:8003" and "audio-fine-tuning:8005" long after both
# were split and renamed.
BACKENDS := $(shell awk '/^  [a-z][a-z0-9-]*:$$/ {svc=substr($$1,1,length($$1)-1)} /^      - "80[0-9][0-9]:8000"$$/ {gsub(/[" ]/,"",$$2); split($$2,p,":"); print svc ":" p[1]}' compose/docker-compose.yml)
PORTS := $(shell echo "$(BACKENDS)" | tr ' ' '\n' | cut -d: -f2 | sort)

up: ## Start every backend + the frontend
	docker compose -f compose/docker-compose.yml up -d
	@echo "Frontend: http://localhost:8500"
	@echo "Backends: $(BACKENDS)"

index-retrieval: ## One-time: populate this stack's own (independent, starts-empty) vector-search-retrieval redis index
	docker compose -f compose/docker-compose.yml exec vector-search-retrieval python -m retrieval.save_doc_embeddings_to_redis

helm-index-retrieval: ## One-time: populate the k8s cluster's own (independent, starts-empty) vector-search-retrieval redis index
	kubectl exec -n $(HELM_NAMESPACE) deploy/vector-search-retrieval -- python -m retrieval.save_doc_embeddings_to_redis

down: ## Stop and remove all containers
	docker compose -f compose/docker-compose.yml down

logs: ## Tail logs from every service
	docker compose -f compose/docker-compose.yml logs -f

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
IMAGES := $(shell grep -oE 'roshb/[a-z0-9.-]+' compose/docker-compose.yml | sort -u)

pin-images: ## Repoint compose and the chart at a published tag: make pin-images TAG=sha-abc123
	@test -n "$(TAG)" || (echo "Usage: make pin-images TAG=sha-<commit>   (CI prints it, or check Docker Hub)" && exit 1)
	@sed -i '' -E 's|(image: roshb/ai-residency-[a-z0-9-]+):\$$\{IMAGE_TAG:-[^}]*\}|\1:$$\{IMAGE_TAG:-$(TAG)\}|' compose/docker-compose.yml
	@sed -i '' -E 's|^  tag: .*|  tag: $(TAG)|' helm/ai-residency/values.yaml
	@echo "Pinned compose and the chart to $(TAG)"

load-images: ## Import locally-built images into the cluster node's own image store
	@for img in $(IMAGES); do \
		docker image inspect $$img:$(IMAGE_TAG) >/dev/null 2>&1 || { echo "skip  $$img:$(IMAGE_TAG) (not built locally)"; continue; }; \
		printf 'load  %s:%s\n' "$$img" "$(IMAGE_TAG)"; \
		docker save $$img:$(IMAGE_TAG) | docker exec -i $(K8S_NODE) ctr -n k8s.io images import - >/dev/null; \
	done
	@echo "Roll the pods to pick them up: kubectl rollout restart deployment -n $(HELM_NAMESPACE)"

helm-install: ## Install the Helm chart onto the local Kubernetes cluster
	helm install $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) --create-namespace $(HELM_SECRETS)

helm-upgrade: ## Re-apply the Helm chart after repulling images, then roll pods to pick them up
	helm upgrade $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) $(HELM_SECRETS)
	kubectl rollout restart deployment -n $(HELM_NAMESPACE)

helm-uninstall: ## Remove the Helm release (namespace itself is left in place, not deleted)
	helm uninstall $(HELM_RELEASE) -n $(HELM_NAMESPACE)
	# Wait for the resources to actually go. `helm uninstall` returns before the
	# API server has finished deleting them, so an install started immediately
	# after can have its own objects deleted by the tail of the previous release
	# -- which is how a reinstall once came up with every pod running and the
	# frontend Service simply absent.
	@kubectl wait --for=delete pod --all -n $(HELM_NAMESPACE) --timeout=120s 2>/dev/null || true
	@kubectl wait --for=delete svc --all -n $(HELM_NAMESPACE) --timeout=60s 2>/dev/null || true

helm-status: ## Show pod/service status for the Helm release
	kubectl get pods,svc -n $(HELM_NAMESPACE)

helm-train: ## Run a training Job on the cluster: make helm-train SERVICE=vit-classifier [EPOCHS=1]
	helm upgrade $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) $(HELM_SECRETS) \
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
