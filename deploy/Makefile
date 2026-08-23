.PHONY: pull up down logs status index-retrieval load-images \
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
HELM_SECRETS := --set-string secrets.hfToken=$(HF_TOKEN) \
                --set-string secrets.wandbApiKey=$(WANDB_API_KEY)
# Suffixes each training Job's name. A Job's pod template is immutable, so a second
# run needs a new name; the epoch second is the cheapest thing that is always unique.
TRAIN_RUN ?= $(shell date +%s)

# Every image is pulled from Docker Hub (roshbeed/ai-residency-<service>-serve / -train),
# built and pushed from the repository root with `make build SERVICE=<name>` and
# `make push SERVICE=<name>`. This directory only deploys them.

pull: ## Pull every backend + frontend image from Docker Hub
	docker compose -f compose/docker-compose.yml pull

up: ## Start every backend + the frontend
	docker compose -f compose/docker-compose.yml up -d
	@echo "Frontend: http://localhost:8500"
	@echo "Backends: word-embeddings:8001 content-scoring:8007 vector-search-retrieval:8002 vision-transformers:8003 multimodal-captioning:8004 audio-fine-tuning:8005 rlhf-ppo:8006"

index-retrieval: ## One-time: populate this stack's own (independent, starts-empty) vector-search-retrieval redis index
	docker compose -f compose/docker-compose.yml exec vector-search-retrieval python -m retrieval.save_doc_embeddings_to_redis

helm-index-retrieval: ## One-time: populate the k8s cluster's own (independent, starts-empty) vector-search-retrieval redis index
	kubectl exec -n $(HELM_NAMESPACE) deploy/vector-search-retrieval -- python -m retrieval.save_doc_embeddings_to_redis

down: ## Stop and remove all containers
	docker compose -f compose/docker-compose.yml down

logs: ## Tail logs from every service
	docker compose -f compose/docker-compose.yml logs -f

status: ## Show which backends are currently reachable (via /health)
	@for w in 8001 8007 8002 8003 8004 8005 8006; do \
		curl -s -o /dev/null -w "port $$w: %{http_code}\n" http://localhost:$$w/health || echo "port $$w: unreachable"; \
	done

# Docker Desktop runs its Kubernetes node as a kind container with its own containerd
# image store. A `docker build` on the host reaches dockerd and stops there, so the
# cluster never sees it -- and because pullPolicy is IfNotPresent, it silently keeps
# running whatever image the node already had. No error, no event, nothing in
# `kubectl describe` to suggest the image is months old. Run this after every rebuild.
K8S_NODE ?= $(shell kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
IMAGE_TAG ?= latest
IMAGES := $(shell grep -oE 'roshbeed/[a-z0-9.-]+' compose/docker-compose.yml | sort -u)

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

helm-status: ## Show pod/service status for the Helm release
	kubectl get pods,svc -n $(HELM_NAMESPACE)

helm-train: ## Run a training Job on the cluster: make helm-train SERVICE=vision-transformers [EPOCHS=1]
	helm upgrade $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) $(HELM_SECRETS) \
		--set train.enabled=true \
		--set train.runName=$(TRAIN_RUN) \
		--set train.only=$(SERVICE) \
		--set train.env.AI_RESIDENCY_NUM_EPOCHS=$(or $(EPOCHS),1)
	@echo "Jobs launched. Follow with: make helm-train-logs"

helm-train-logs: ## Tail whichever training Jobs are currently running
	kubectl get jobs -n $(HELM_NAMESPACE)
	kubectl logs -n $(HELM_NAMESPACE) -l app.kubernetes.io/name --tail=40 --prefix \
		--selector='app.kubernetes.io/name in (word-embeddings-train,content-scoring-train,vector-search-retrieval-train,vision-transformers-train,multimodal-captioning-train,audio-fine-tuning-train,rlhf-ppo-train)'

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
