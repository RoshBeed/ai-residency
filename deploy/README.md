# Deployment

How the portfolio runs: a docker-compose stack for a single `docker compose up`, and a Helm chart
for a real Kubernetes cluster. Same ten containers either way — nine backends, the frontend, and
the Redis the retrieval service indexes into.

Nothing here builds from source. It consumes images built at the repository root
(`make build SERVICE=<name>`), so the deployment description and the code that produced it move
together in one commit.

| | |
|---|---|
| [`../services/`](../services) | The services this deploys, one directory each |
| [`../libs/common`](../libs/common) | Shared helpers every backend depends on |
| [`../services/frontend`](../services/frontend) | The Streamlit frontend that calls them |

## Layout

```
deploy/
├── compose/docker-compose.yml   every backend + Redis + the frontend, one command
├── helm/ai-residency/           the same services as a Helm chart
├── terraform/aws-training/      AWS spot GPU training, an alternative to RunPod
├── Makefile                     up / down / status / load-images / helm-* targets
└── .env.example                 HF_TOKEN, WANDB_API_KEY (optional — both degrade gracefully)
```

## Images come from Docker Hub

Every container is pulled as `roshb/ai-residency-<service>-serve:latest` (or `-train`, or
`roshb/ai-residency-frontend`). They are built and published from the repository root:

```bash
make build SERVICE=word-embeddings
make push  SERVICE=word-embeddings
```

Serving images are published by CI on every push to `main`, so `docker compose pull` fetches them.
Training images are published on demand by the `Publish training images` workflow. To build locally
instead — the same names compose and the chart already reference:

```bash
cd ..
make base        # roshb/ai-residency-base-serve (CPU)
make base-train  # roshb/ai-residency-base-train (CUDA)
for s in $(ls -d services/*/ | xargs -n1 basename | grep -v _template | grep -v frontend); do
  make build SERVICE=$s
done
docker build -f services/frontend/Dockerfile.serve -t roshb/ai-residency-frontend:latest .
```

The base image comes first because every `Dockerfile.serve` builds `FROM` it.

That is enough for docker-compose, which reads the host's images directly. **Kubernetes does
not.** Docker Desktop runs its node as a kind container with its own containerd image store, so
a `docker build` on the host is invisible to the cluster — and since `imagePullPolicy` is
`IfNotPresent`, the node keeps running whatever image it already had, with no error and nothing
in `kubectl describe` to say the image is stale. After building, import them:

```bash
make load-images          # docker save | ctr -n k8s.io images import, per image
kubectl rollout restart deployment -n ai-residency
```

Trained checkpoints are a separate concern from the images. Each backend mounts
`../services/<name>/checkpoints` as the fast local path, and also
falls back to pulling weights from the Hugging Face Hub when the mount is empty or absent (see
`libs/common`'s `resolve_checkpoint`). It reports which happened via `checkpoint_source` on
`/health`.

## Quickstart

```bash
docker compose pull       # pull every image from Docker Hub (once pushed)
docker compose up -d
```

Then open http://localhost:8500. Each backend is also reachable directly for debugging:
word-embeddings `:8001`, content-scoring `:8007`, then vector-search-retrieval `:8002` … rlhf-ppo `:8006`.
`make status` curls each one's `/health`, which all seven expose.

**One-time step for vector-search-retrieval:** this stack runs its own independent Redis (`redis`), separate
from — and starting empty, unlike — the original vector-search-retrieval repo's own persistent one. Populate it once
after the first `up`:

```bash
docker compose exec vector-search-retrieval uv run python -m retrieval.save_doc_embeddings_to_redis
```

Without this, vector-search-retrieval's `/query` page returns an empty result set (not an error — just nothing to
retrieve yet).

All seven backends answering `/health` under compose, and the frontend serving the portfolio
at `:8500`:

![The portfolio frontend, one card per system, served by the docker-compose stack](assets/compose-frontend.png)

**Known constraint, confirmed empirically, not just theoretical:** Docker Desktop's VM has a fixed
memory ceiling (~7.65GB on the machine this was built on — check `docker stats`/Docker Desktop's
resource settings on yours). Running all the backends for an extended period eventually OOM-kills
one of the heavier ones (observed here: multimodal-captioning after ~25 hours, rlhf-ppo shortly after that) —
`docker compose ps -a` will show `Exited (137)` for whichever one got killed. This isn't a bug in
this repo — it's Docker Desktop's own memory ceiling. Fix: `docker compose up -d weekN` restarts
just that one; give Docker Desktop more memory in its settings if you want long sustained uptime
for the full stack; or don't run all of them for days at a stretch on a constrained host.

**Checkpoint mounts are writable on purpose.** The Hub fallback downloads into the same directory
it looked in, so mounting `checkpoints` read-only turns "fall back to the Hub" into a guaranteed
`[Errno 30] Read-only file system` on exactly the hosts that have no local copy.

## Kubernetes (Helm)

`helm/ai-residency/` is the Kubernetes analog of `docker-compose.yml` — the same ten containers
(nine backends + frontend) plus the retrieval service's Redis, expressed as Deployments/Services instead of compose
services.

```bash
make helm-install         # helm install ai-residency ./helm/ai-residency -n ai-residency --create-namespace
```

Then open http://localhost:8500 — Docker Desktop's Kubernetes auto-binds `LoadBalancer` services to
localhost, so the frontend Service is reachable with no port-forward needed. The backends are
deliberately `ClusterIP`-only (mirrors the real shape: only the frontend talks to them directly) —
reach one directly for debugging with `kubectl port-forward -n ai-residency svc/word-embeddings 8001:8000`
(repeat per backend, ports 8001–8007 to match compose's numbering). Same one-time step as the compose
version to populate vector-search-retrieval's (independent, starts-empty) Redis index:

```bash
kubectl exec -n ai-residency deploy/vector-search-retrieval -- uv run python -m retrieval.save_doc_embeddings_to_redis
```

`make helm-upgrade` re-applies the chart *and* runs `kubectl rollout restart deployment` afterward —
needed because a fixed `latest` tag means the manifest itself never changes when the underlying
image is rebuilt, so `helm upgrade` alone wouldn't actually roll already-running pods onto a freshly
pulled image. `make helm-uninstall` tears the release down (the namespace is left in place);
`make helm-status` shows pod/service state.

**`hostPath` volumes — local-node convenience, not portable:** every volume mount in `values.yaml`
points at an absolute path on the *node's* filesystem (`hostRoot: /Users/rosh/mlx` by default). This
only works because Docker Desktop's Kubernetes node's filesystem **is** the Mac's filesystem. It
reproduces docker-compose.yml's own mounts. The gitignored-artifact mounts (trained checkpoints,
which may legitimately be absent on a host that never trained/downloaded them) use
`DirectoryOrCreate`/`FileOrCreate` hostPath types instead of stricter `Directory`/`File` — a strict
type fails the pod's volume mount outright when the path doesn't exist yet (`FailedMount`, stuck
`ContainerCreating`), which would prevent the app's own graceful missing-checkpoint handling from
ever running. A real multi-node cluster would need actual artifact hosting (git-lfs, release
assets, DVC) in place of these hostPath mounts, or — better — would drop them entirely and let the
The Hugging Face model cache is a **separate** value (`hfCachePath`), not derived from `hostRoot`:
it lives under the user's home directory, outside the portfolio checkout. Deriving it from
`hostRoot` mounts a path that does not exist, and because these volumes use `DirectoryOrCreate`,
Kubernetes silently creates an empty one — every pod then re-downloads several GB of weights on
each cold start, with no error to notice. Compose expresses the same path as
`${HOME}/.cache/huggingface`.

If this repo ever moves to a different path, update `hostRoot` in
`values.yaml` (or pass `--set hostRoot=...` at install time) rather than editing each mount.

**Health probes gate traffic.** Every backend exposes `/health`, and the chart wires all three
probe types to it (`probes` in `values.yaml`). This matters more than it looks: without a
readiness probe Kubernetes adds a pod to its Service's endpoints the moment the container process
exists — minutes before the model has finished loading — so the frontend's first requests get
routed to a backend that cannot answer them. The startup probe carries a deliberately generous
budget (up to 10 minutes) because a cold start may pull weights from the Hub, and Kubernetes
suppresses the liveness probe while it runs, so a slow-but-healthy load is never mistaken for a
hang and restarted in a loop.

**Deployments use `strategy: Recreate`, not the default `RollingUpdate`.** A rolling update starts
the replacement pod before retiring the old one, so the node needs two copies' worth of memory at
once. These pods request 1.5–6GiB each and the full set already sits near a single node's
allocatable, so the surge pod stays `Pending` — and since the old pod is only retired once the new
one is Ready, the rollout deadlocks instead of finishing (observed on `make helm-upgrade`: vector-search-retrieval,
multimodal-captioning and rlhf-ppo all stuck on `Insufficient memory`). A few seconds of downtime per deploy is the
right trade here. On a real multi-node cluster, drop it. Note that switching an existing release to
`Recreate` needs a reinstall, since the live Deployment still carries `rollingUpdate` fields.

**Memory ceiling — confirmed empirically, worse under Kubernetes than under compose:** under
compose an OOM-killed container just stays stopped, but under Kubernetes a Deployment's default
`restartPolicy: Always` immediately recreates the pod, which re-competes for the same scarce
memory and is killed again — confirmed on first install: 8 pods starting simultaneously with no
resource requests produced an infinite crash loop across the whole namespace. Every container now
sets `resources.requests.memory` and `resources.limits.memory` (`memoryRequest`/`memoryLimit` in
`values.yaml`), so the scheduler refuses to place what does not fit and a pod sits `Pending`
instead of the set thrashing.

The requests total roughly 17.5GiB, so whether all nine pods fit depends entirely on how much
memory Docker Desktop is given — its default leaves the node well short, and some pods stay
`Pending` until it is raised. Read the real budget rather than trusting a number here:

```bash
kubectl get node -o jsonpath='{.items[0].status.allocatable.memory}'
```

At 23.4GiB allocatable all nine schedule and stay `Running` with zero restarts. Otherwise, either
raise Docker Desktop's memory (the VM restart bounces every pod, and one that was already
`Pending` beforehand may need a `kubectl delete pod <name>` nudge to be re-evaluated) or free room
with `kubectl scale deploy/multimodal-captioning -n ai-residency --replicas=0`.

Real inference through the cluster — the frontend on the `LoadBalancer`, calling word-embeddings'
`ClusterIP` Service, which resolved its checkpoint and returned genuine nearest neighbours:

![Word Embeddings page served from Kubernetes, returning real nearest neighbours for "python" with cosine scores](assets/kubernetes-inference.png)

**If anything is also running outside the cluster, stop it first — confirmed to actually happen:**
this repo's own `docker compose up` stack, *and* any individual week repo's own standalone
`docker compose up` (e.g. Week 3's own `serve` containers, left over from verifying that repo) both
stand up real, heavy containers that compete for the exact same memory budget as the cluster.
`docker ps` shows Kubernetes-managed containers with a `k8s_<container>_<pod>_<namespace>_...` name
prefix — anything without that prefix is running outside the cluster. `docker compose down` (from
this repo, or the individual week repo) clears it.

## Deploying to real hosts (RunPod / AWS / Hetzner)

The portfolio is built to split across two machines that share no filesystem: a GPU box that trains
and a cheap VM that serves. What connects them is the Hugging Face Hub — training pushes a
checkpoint, serving pulls it. Neither box needs anything from this Mac. This was rehearsed locally
before spending anything on it: every backend was started with its checkpoint deliberately hidden
and confirmed to pull from the Hub and return identical output.

### Credentials

Both boxes read a `.env` (see `.env.example`). `libs/common` loads it on import, walking up from the
working directory, so one file beside several repos covers all of them.

```bash
HF_TOKEN=hf_...          # write on the training box, read on the inference box
WANDB_API_KEY=...        # training box only; without it runs go offline, not missing
```

**The model repos are private.** An inference box without `HF_TOKEN` will not fail loudly — it will
warn, serve an untrained model, and report `checkpoint_loaded: false`. Check `/health` after a
deploy rather than assuming a 200 means it is working.

### Training box

```bash
git clone <week repo> && cd <week repo>
cp .env.example .env      # fill in both tokens
uv sync
AI_RESIDENCY_NUM_EPOCHS=50 make train
```

Any config field is overridable as `AI_RESIDENCY_<FIELD_NAME>`, so a real run needs no code edit. On a new
best checkpoint the run saves locally, pushes to the Hub, and logs to W&B; a failed push is logged
and does **not** end the run.

Per-repo caveats worth knowing before you rent anything:

| Repo | Caveat |
|---|---|
| Week 1 — word embeddings | `train-corpus` needs a Hacker News Postgres mirror that is not publicly reachable; `train-cbow` works from data on the Hub. |
| Week 1 — content scoring | Trains only the regression head; pulls the frozen CBOW vectors from the word-embeddings repo's Hub checkpoint. |
| Week 2 | Needs Redis for serving (not for training). `make serve` is the HTTP service; `make query` is an interactive CLI. |
| Week 3 | Two trainers with separate epoch caps — the classifier reads `AI_RESIDENCY_CLASSIFIER_EPOCHS`, not `AI_RESIDENCY_NUM_EPOCHS`. |
| Week 4 | `make train` overwrites a **committed** adapter. Set `AI_RESIDENCY_CHECKPOINT_NAME` to something else. Pins itself to CPU: MPS training here was non-reproducible. |
| Week 5 | Needs the **system** `ffmpeg` binary, not the pip package of the same name. |
| Week 6 | One PPO step, deliberately — there is no epoch loop, because nothing here trains the reward model. |

### AWS EC2 spot GPU — alternative to RunPod

`infra/aws-training/` is Terraform that provisions a single AWS spot GPU instance, pulls a pushed
`roshb/ai-residency-<service>` image (built by that week's own `Dockerfile.train`,
`make push SERVICE=<name>`), runs it, and terminates itself when training finishes or times out. It
moved here from Week 3 (where it started as a Week-3-only script) because it isn't Week-3-specific
infrastructure — `train_image` picks which week's already-built image to run.

```bash
cd infra/aws-training
cp terraform.tfvars.example terraform.tfvars   # at minimum: train_image, hf_token
terraform init
terraform plan      # review before spending anything
terraform apply     # provisions a real, billable spot instance
```

Or `./quick-start.sh` for interactive setup, `./monitor.sh` to check on a running instance,
`./cleanup.sh` to tear it down manually. Full detail — including two real bugs found while
generalizing this out of Week 3 (a Terraform template referencing undefined variables, and a
duplicate output definition — either alone means the original could never have passed
`terraform validate`) — is in `infra/aws-training/README.md`.

### Inference box

```bash
git clone <week repo> && cd <week repo>
cp .env.example .env      # HF_TOKEN alone is enough
uv sync
make serve
curl localhost:8000/health
```

No checkpoint needs copying. The first request resolves it: local file if present, Hub otherwise.
Expect `"checkpoint_source": "hub"` on a fresh box — that is the whole mechanism working. A
container works the same way, with no volume at all: `docker pull roshb/ai-residency-<service>:latest`
then `docker run -e HF_TOKEN="$HF_TOKEN" -p 8000:8000 roshb/ai-residency-<service>:latest`.

## Related

The frontend and the full documentation for all weeks live in `services/frontend`. The
backend repos are container source only: two Dockerfiles (`Dockerfile.train` for a RunPod/AWS
GPU host, `Dockerfile.serve` for Kubernetes) and a short technical README:

[Week 1 — word embeddings](https://github.com/RoshBeed/ai-residency/tree/main/services/word-embeddings) ·
[Week 1 — content scoring](https://github.com/RoshBeed/ai-residency/tree/main/services/content-scoring) ·
[Week 2](https://github.com/RoshBeed/ai-residency/tree/main/services/vector-search-retrieval) ·
[Week 3](https://github.com/RoshBeed/ai-residency/tree/main/services/vision-transformers) ·
[Week 4](https://github.com/RoshBeed/ai-residency/tree/main/services/multimodal-captioning) ·
[Week 5](https://github.com/RoshBeed/ai-residency/tree/main/services/audio-fine-tuning) ·
[Week 6](https://github.com/RoshBeed/ai-residency/tree/main/services/rlhf-ppo)
