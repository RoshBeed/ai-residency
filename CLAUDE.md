# AI Residency — working notes

Nine ML systems, each from scratch to a containerized service, in one uv workspace. Originally
projects from the [Machine Learning Institute](https://ml.institute/) residency; now a portfolio
repo. Private, at `github.com/RoshBeed/ai-residency`.

## Layout

```
libs/common/                 device, logging, checkpoints, config, seeding, Hub, W&B — used by all
libs/vision-transformer/     attention, encoder, decoder, patch embedding — used by the two ViTs
services/<name>/             one per system: package, config/, serving/, tests/, Dockerfiles, README
services/_template/          the shape a new service starts from; a working one, not a skeleton
services/frontend/           the Streamlit app, one page per system
deploy/{compose,helm,terraform}/
notebooks/<service>/         chdir into ../../services/<service> when run
```

Services: `word-embeddings`, `content-scoring`, `vector-search-retrieval`, `vit-classifier`,
`vit-seq2seq`, `multimodal-captioning`, `whisper-fine-tuning`, `speech-emotion-recognition`,
`rlhf-ppo`.

## Working in it

```bash
make install                      # one venv for the workspace
make check                        # ruff + strict mypy + pytest, every service
make check SERVICE=<name>         # just one
make build SERVICE=<name>         # both images, built from the repo root context
make -C services/<name> help      # that service's own pipeline stages
```

- **mypy runs per service, deliberately.** Each has its own top-level `config` package, so one
  invocation over the tree sees nine modules named `config` and refuses to start.
- **Images build from the repository root**, not the service directory — they need the root
  `pyproject.toml`, the shared `uv.lock` and `libs/`.
- **After adding or renaming a workspace member**, run `uv lock`. The Dockerfiles pick it up on
  their own: they copy manifests with `COPY --parents libs/*/... services/*/...`, so there is no
  per-member list to go stale.

**Two registries, two usernames.** Docker Hub is **`roshb`** (image names only); Hugging Face is
**`roshbeed`** (`hf_repo_id`, `hf_dataset_repo_id`, hub URLs). They are not interchangeable, and a
blanket find-and-replace across `roshbeed/` will silently repoint every checkpoint at a Hugging
Face account that does not exist.

## Standing rules

**No local checkpoints, ever.** Every runtime is ephemeral. Training pushes to the Hugging Face
Hub, serving pulls from it. Resolve through `ai_residency_common`'s `resolve_checkpoint` /
`resolve_required` / `load_model`, never a bare `Path.exists()`. Do not add checkpoint mounts to
compose or `hostPath` volumes to the chart. Uploads need a stable, guessable filename — a
run-timestamped name is unfetchable by anything downstream.

**Structure first; model quality later.** The current phase is that everything is plumbed
correctly end to end. Untrained or weakly-trained checkpoints are expected, not a defect. A
service that starts, reports `checkpoint_loaded: false` and answers 503 honestly is *passing*.
Verify wiring, not accuracy; don't propose training runs as the next step.

**Verify by running, not reading.** Nearly every real bug in this repo was found by making an
actual request, not by inspecting code — a service that builds, starts and passes `/health` can
still 500 on the first real call. Health checks are not verification.

**Images are published to Docker Hub publicly** (the free tier allows one private repository, and
there are twelve images). CI pushes the serving base and the nine `-serve` images on every push to
`main`, gated behind the `PUBLISH_IMAGES` repository variable and the `DOCKERHUB_TOKEN` secret.

Training images go through the separate `Publish training images` workflow, run on demand: they are
~9GB each, and the base plus one service does not fit in a runner's default free disk, so that
workflow clears the preinstalled toolchains and builds one service per job. Runners are x86-64, so
those builds are native rather than the slow emulation a Mac would do.

## Environment gotchas that still bite

- **The Kubernetes node has its own image store.** Docker Desktop runs it as a kind container, so
  `docker build` on the host is invisible to the cluster and `imagePullPolicy: IfNotPresent`
  silently keeps a stale image with no error anywhere. `make -C deploy load-images` after every
  rebuild.
- **`datasets/` means two different things.** Raw corpora in the audio services, a Python package
  in the vision ones. Both `.gitignore` and `.dockerignore` exclude only the audio ones; a blanket
  rule silently drops source and produces an image that builds fine and fails at import.
- **Two base images, one Dockerfile.** `make base` builds `ai-residency-base-serve` (CPU torch);
  `make base-train` builds `ai-residency-base-train` (CUDA, `linux/amd64`). Same `Dockerfile.base`,
  different build args. Serving images build `FROM` the first, training images `FROM` the second.
- **Downstream images sync with the torch stack excluded**:
  `uv sync --frozen --no-install-package torch --no-install-package torchvision
  --no-install-package torchaudio --inexact`. Without that, uv reinstalls the lock's CPU wheels over
  a CUDA base; `--inexact` stops it removing them as extraneous. `grep -c 'name = "nvidia-' uv.lock`
  must stay 0 — the lock is CPU-only by design, and CUDA comes from the base, not from uv.
- **Training images must target the GPU host's architecture.** `make build` passes
  `--platform linux/amd64` for them. Built on an Apple Silicon Mac without it they come out
  linux/arm64 and will not run on an x86-64 instance.
- **`helm uninstall` returns before deletion finishes.** An install started right after can have
  its objects deleted by the tail of the previous release — a reinstall once came up with all
  eleven pods running and the frontend Service simply absent. `make helm-uninstall` now waits.
- **`.python-version` pins 3.11.** Without it uv picks per-host and mypy rejects numpy's stubs.
- **`AI_RESIDENCY_<FIELD>` overrides any `Config` field** — `AI_RESIDENCY_NUM_EPOCHS=1` for a smoke
  run. `vit-classifier` uses `classifier_epochs`, so it needs `AI_RESIDENCY_CLASSIFIER_EPOCHS`.
- **Docker Desktop's VM goes read-only when the host disk fills**, and stays read-only until the VM
  itself restarts — quitting the app is not enough unless `com.docker.virtualization` actually exits.

## Writing conventions

Service READMEs are the write-up *and* the GitHub landing page, and the frontend links to them.

- Write like an engineer describing a system they built. First person for real decisions ("I went
  with late fusion because…"), "this service" for what it does.
- **Skimmable.** Short paragraphs, one idea each. No wall of prose with inline code embedded in it.
- **Don't dwell on what isn't trained.** The architecture and the plumbing are the point. State a
  limitation once, forward-looking, and move on. Don't write bug diaries or "what went wrong"
  sections.
- Cut AI tells: rhetorical-question openers, "exactly" as a tic, templated transitions, heavy
  em-dash use. Never cut the *why* behind a technique — that reasoning is the point.
- Real numbers only, always next to their baseline. Cite papers; course material needs no citation.
- Explain a shared thing once, in the library that owns it, and link to it.
- Validate every Mermaid block: `npx -y @mermaid-js/mermaid-cli -i b.mmd -o b.svg`.

## State

CI is one path-filtered workflow: lint, then per-service mypy and pytest, then a publish job gated
behind a `PUBLISH_IMAGES` repository variable. 147 tests.

Open:
- `services/frontend` has no tests. `app.py` runs its whole page flow at import, so its display
  helpers cannot be imported without a Streamlit runtime; extracting them into a module is the fix,
  and worth doing — strict typing already found a real bytes/str bug in there. CI warns rather than
  failing on a service with no `tests/`.
- `speech-emotion-recognition` has no trained checkpoint, so it answers 503. Its pipeline, trainer
  and API are wired and verified.
- `rlhf-ppo` has one `xfail`: the PPO ratio can differ for one element because the generator
  records the state as decoded text and the ratio re-tokenizes it, and detokenize/retokenize is not
  always a round-trip. Carrying token ids through is the fix — PPO internals, so it waits.
- The eleven predecessor repos are archived on GitHub. `backup/pre-squash-2026-08-23` tags the
  pre-history-rewrite tip; delete it once the current history is settled.
