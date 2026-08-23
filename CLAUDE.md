# AI Residency — working notes

Nine ML systems, each from scratch to a containerized service, in one uv workspace. Originally
projects from the [Machine Learning Institute](https://ml.institute/) residency; now a portfolio
repo. Private, at `github.com/RoshBeed/ai-residency`.

## Layout

```
libs/common/                 device, logging, checkpoints, config, seeding, Hub, W&B — used by all
libs/vision-transformer/     attention, encoder, decoder, patch embedding — used by the two ViTs
services/<name>/             one per system: package, config/, serving/, tests/, Dockerfiles, README
services/_template/          what `make new-service NAME=x` copies; a working service, not a skeleton
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

**Nothing is published to Docker Hub** while the repo is private: the free tier allows one private
repository, so publishing would make the source publicly pullable ahead of the repo. Build locally
and use `make -C deploy load-images`. Revisit, along with pinning `deploy/` to a commit SHA
instead of `:latest`, when the repo goes public.

## Environment gotchas that still bite

- **The Kubernetes node has its own image store.** Docker Desktop runs it as a kind container, so
  `docker build` on the host is invisible to the cluster and `imagePullPolicy: IfNotPresent`
  silently keeps a stale image with no error anywhere. `make -C deploy load-images` after every
  rebuild.
- **`datasets/` means two different things.** Raw corpora in the audio services, a Python package
  in the vision ones. Both `.gitignore` and `.dockerignore` exclude only the audio ones; a blanket
  rule silently drops source and produces an image that builds fine and fails at import.
- **Check the lock for CUDA after any `uv lock`**: `grep -c 'name = "nvidia-' uv.lock` must be 0.
  A newer torch re-resolves the CUDA build and bloats every image several-fold.
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
- `speech-emotion-recognition` has no trained checkpoint, so it answers 503. Its pipeline, trainer
  and API are wired and verified.
- `rlhf-ppo` has one `xfail`: the PPO ratio can differ for one element because the generator
  records the state as decoded text and the ratio re-tokenizes it, and detokenize/retokenize is not
  always a round-trip. Carrying token ids through is the fix — PPO internals, so it waits.
- The eleven predecessor repos are archived on GitHub. `backup/pre-squash-2026-08-23` tags the
  pre-history-rewrite tip; delete it once the current history is settled.
