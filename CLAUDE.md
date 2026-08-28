# AI Residency — working notes

Nine ML systems, each from scratch to a containerized service, in one uv workspace. Originally
projects from the [Machine Learning Institute](https://ml.institute/) residency; now a portfolio
repo. Private, at `github.com/RoshBeed/ai-residency`.

## Layout

```
libs/common/                 device, logging, checkpoints, config, seeding, Hub, W&B, health/ready
libs/vision-transformer/     attention, encoder, decoder, patch embedding — the two ViTs
libs/word2vec/               CBOW and skip-gram, negative sampling or softmax, nearest
                             neighbours — word-embeddings, vector-search-retrieval
libs/speech/                 Whisper loading, features, labels, decoding — the two audio services
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

**One implementation per algorithm.** Three services had grown their own word2vec and two
their own Whisper layer, because each week's team either started fresh or picked up the
previous team's code. Those are `libs/word2vec` and `libs/speech` now. When you find a fourth
copy of something, the answer is a library, not a tidy-up of one copy — the git history keeps
the provenance either way.

**Vocabulary building is deliberately not shared, and that was checked rather than assumed.**
`word-embeddings/etl.py` (947 lines) and `vector-search-retrieval/cbow/train_tokenizer.py`
(117) both end in a word→id mapping, which looks like the same duplication word2vec was. It
is not. The first normalises real-world Unicode, pulls from a Hacker News Postgres mirror and
a Wikipedia dump, merges corpora and lemmatises; the second lowercases text8, which is already
ASCII, and counts. The genuinely common part is about twenty lines — a `Counter`, a sort, and
a reserved `<PAD>` at id 0 — and extracting that would buy less than the indirection costs.
Two implementations of *the same algorithm* is a library; two pipelines that happen to end in
a dict is not.

**Two registries, two usernames.** Docker Hub is **`roshb`** (image names only); Hugging Face is
**`roshbeed`** (`hf_repo_id`, `hf_dataset_repo_id`, hub URLs). They are not interchangeable, and a
blanket find-and-replace across `roshbeed/` will silently repoint every checkpoint at a Hugging
Face account that does not exist.

## Standing rules

**Nothing depends on someone else's account.** Weights, datasets and the text8 corpus are all
under `roshbeed/` on the Hub. text8 used to be fetched from a teammate's personal repo, which is
a single point of failure outside this project's control — it is mirrored at
`roshbeed/ai-residency-text8`, verified byte-identical to the canonical source at
mattmahoney.net. Re-hosting a *standard public corpus* is right; the earlier "don't re-host
third-party data" note was about not copying someone's proprietary dataset, and it got misapplied.

**No local checkpoints, ever.** Every runtime is ephemeral. Training pushes to the Hugging Face
Hub, serving pulls from it. **The model and dataset repos are public**, so serving needs no
credentials at all — `HF_TOKEN` is only for pushing. Resolve through `ai_residency_common`'s `resolve_checkpoint` /
`resolve_required` / `load_model`, never a bare `Path.exists()`. Do not add checkpoint mounts to
compose or `hostPath` volumes to the chart. Uploads need a stable, guessable filename — a
run-timestamped name is unfetchable by anything downstream.

**Structure first; model quality later.** The current phase is that everything is plumbed
correctly end to end. Untrained or weakly-trained checkpoints are expected, not a defect. A
service that starts, reports `checkpoint_loaded: false` and answers 503 honestly is *passing*.
Verify wiring, not accuracy; don't propose training runs as the next step.

**`/health` and `/ready` are different endpoints, and must stay different.** `/health` is
liveness — always 200, so a slow or missing checkpoint never restarts a pod. `/ready` is
readiness — 503 until the service can actually serve, so an unusable pod is kept out of its
Service's endpoints. Both come from `add_health_routes` in `ai_residency_common.serving`; never
hand-write a health endpoint in a service. Pointing a readiness probe at `/health` gives you a
probe that cannot fail, which is the same as having none.

**The UI lives in one container, never in a backend.** Backends are HTTP APIs and
nothing else — no Streamlit, no templates, no static pages. The frontend image is the
only place Streamlit runs, and `AI_RESIDENCY_PAGES` tells it which pages to show, so
the same image is the nine-system hub or the UI for one system. `make up
SERVICES="..."` sets it from what it started. Putting a page in a backend would mean
either Streamlit in all nine images or a second UI codebase in HTML/JS, and the
per-service `/docs` FastAPI already generates covers the "just poke this one service"
case for free.

**Every numeric request field is bounded.** A field that reaches a model or a
datastore is an unbounded request unless `Field(ge=…, le=…)` says otherwise, and the
bound belongs on the request model so validation rejects it before the work starts.
Two measured examples of what "unbounded" costs: `max_new_tokens=100000` on
`multimodal-captioning` was still generating after 60 seconds and holding a worker,
where the default of 30 answers in about two; and `top_k=-1` on
`vector-search-retrieval` made Redis reject the query, which the handler reported as
`503 "Redis unreachable"` — blaming the datastore for the caller's input, on a Redis
that was perfectly reachable. Both were reachable *only against a deployed cluster*:
in-process the services 503 without loaded state, so the input never gets near the
model or Redis. Probe the deployed thing, not just the imported app.

**One status code for one class of failure: 422, never 400.** A request that is
well-formed but whose content cannot be processed — undecodable base64, bytes that are
not an image or audio, an empty prompt — answers 422 everywhere. This was settled once
and then drifted three separate times, so it is written down: `decode_image` and
`decode_audio` in `libs/common` are the reference, and a service that catches their
failure must not downgrade it. Guard the *content*, not the envelope — every 500 found
by probing this API surface was a guard that checked the wrapper and trusted what was
inside it. `Image.open` had not read the pixels; the audio decoder raised a
`DecodeError` rather than the `RuntimeError` that was caught.

**Containers run as UID 1000, not root.** `Dockerfile.base` creates the `app` user; every service
Dockerfile chowns its own directory and ends with `USER app`. The chart's
`securityContext.runAsUser` is the same 1000, and it has to be — Kubernetes overrides the image's
USER, so a mismatch produces a container that starts and then cannot write.

**Verify by running, not reading.** Nearly every real bug in this repo was found by making an
actual request, not by inspecting code — a service that builds, starts and passes `/health` can
still 500 on the first real call. Health checks are not verification.

**CI runs every image before it publishes it.** The publish job builds native amd64 into the
runner's docker with `load: true`, smoke-tests it, and only then does the multi-arch push. The
smoke test asserts the container is non-root, that `/health` *and* `/ready` are both registered,
and that the working directory is writable by that user. Those are the three ways a change here
builds green and fails at runtime: a `USER` whose file ownership is wrong, and a dropped route
that turns every readiness probe into a permanent 404. It deliberately does not start the server
— several services load multi-GB backbones in their lifespan, which does not fit a runner — so it
imports the app and reads its route table instead.

**Published images are `linux/amd64` only.** arm64 has to be emulated through QEMU on
an x86 runner, and measured across a full publish it cost **8.4 minutes — 47% of all
build time** — for an architecture nothing in the deployment path runs: the cluster, the
GPU host and the runners are all x86-64. It existed so images ran natively on an Apple
Silicon laptop; `make local SERVICE=<name>` builds natively there instead, base included,
because a service image is only as native as what it builds `FROM`.

**`deploy/` is tested, and not by the member matrix.** `make check` and CI's lint job
run `deploy/` explicitly, because the per-member matrix covers `libs/*` and `services/*`
only — so the tooling that decides what the cluster actually deploys was the one thing in
the repo nothing ran. Bare `python` is not on PATH in every context either: use `uv run
python` in the Makefiles and `python3` on a runner.

**`deploy/images.lock` is the manifest; digests, not a shared tag.** A tag describes
*one image's* content. `deploy/` needed something describing *the whole set*, and
overloading one `image.tag` across eleven images only holds while every image changes on
every commit — the moment one service changes and eight do not, the tag either lies or
has to be minted for images that did not earn it. The lock names each image by immutable
digest; `python deploy/pin_images.py` propagates it into the chart and compose, and
`--check` fails on drift. `git log deploy/images.lock` is the deployment history.

**A local cluster runs locally-built images.** Published images are `linux/amd64` and
Docker Desktop's Kubernetes node is **arm64** on Apple Silicon, so it cannot pull them at
all — `make -C deploy local-cluster` builds natively, loads into the node's own image
store, and installs with `image.useDigest=false` so the chart asks for the tag those
images carry rather than a published digest they will never match. All three steps are
needed and each is easy to forget separately.

**Only changed images are built, and nothing is retagged.** CI's publish matrix
covers `changes.outputs.services`, so an unchanged image gets no build and no tag it
did not earn. This replaced a scheme that rebuilt all eleven every push -- 78% of this
repo's Actions minutes -- and then, briefly, one that retagged the unchanged ones. Both
existed only because `deploy/` pinned a single tag across every image; the manifest
removed the reason for either.

**Images are published to Docker Hub publicly** (the free tier allows one private repository, and
there are twelve images). CI pushes the serving base and *all ten* service images on every push to
`main`, gated behind the `PUBLISH_IMAGES` repository variable and the `DOCKERHUB_TOKEN` secret.
Publishing deliberately ignores the path filter that narrows the checks: one `sha-<commit>` tag has
to name a complete set, or pinning it in `deploy/` describes something the registry cannot serve.

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
- **`helm upgrade --wait` cannot succeed while a service is deliberately unready.**
  `--wait` blocks until every Deployment reports available, and
  `speech-emotion-recognition` has no checkpoint, so its `/ready` is an honest
  permanent 503 and the rollout hits the progress deadline. Helm then marks the
  release `failed` even though all ten pods rolled out on the new tag. That is why
  `make helm-upgrade` does not pass `--wait`; don't add it back until every service
  has a checkpoint.
- **`helm uninstall` returns before deletion finishes.** An install started right after can have
  its objects deleted by the tail of the previous release — a reinstall once came up with all
  eleven pods running and the frontend Service simply absent. `make helm-uninstall` now waits.
- **An embedding matrix can have more rows than the vocabulary has words.** The
  word-embeddings checkpoint is 258,196 rows against 230,701 words, so 27,495 rows
  cannot be named. `topk` searches all of them, a nameless row wins, and the lookup
  after it raises `KeyError` -- `/nearest` returned 500 for any `k` above about 9
  while the schema advertised up to 100. Never index a vocabulary with a raw `topk`
  result; go through `word2vec.nearest_words`, which masks the unnamable rows out of
  the search.
- **The PPO ratio scores token ids, never re-tokenized text.** A generated state is
  carried as `state_token_ids` as well as `state_text`, and `PolicyRatio` reads the ids.
  Re-encoding the text looks equivalent and is not: about one in twenty states does not
  survive decode-then-encode, sometimes coming back with a *different number* of tokens,
  so the two sides of the ratio scored different sequences and the ratio was not 1 before
  any gradient step. The value model still reads `state_text` — that one is genuinely text.
- **Whisper labels must not start with `<|startoftranscript|>`.** The model prepends
  `decoder_start_token_id` itself when it shifts labels right, and that token *is* SOT, so
  passing the tokenizer's output through unchanged trains the decoder one position out of
  step. Nothing errors; the loss simply describes a different task. Use
  `speech.encode_target`, which drops it.
- **`.python-version` pins 3.11.** Without it uv picks per-host and mypy rejects numpy's stubs.
- **buildx resolves a base-image tag against the registry, not the local store.** `make base`
  followed by `make build SERVICE=x` silently built against the *published* base until the Makefile
  started passing `--build-arg BASE_IMAGE=` explicitly. Same class of trap as the kind node's
  separate image store, one layer up.
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

CI is one path-filtered workflow: lint, then per-service mypy and pytest, then build → smoke-test →
publish (all images, not just the changed ones) gated behind a `PUBLISH_IMAGES` repository variable,
then a Trivy scan of the published base whose results go to the job summary (SARIF upload
needs Advanced Security, which a private repo does not have). 282 tests.

Building images locally is for debugging, not delivery — CI is what publishes. `make base` and
`make build` exist so a change to a Dockerfile can be exercised before it is pushed.

`services/frontend` is now four modules plus the page flow: `projects.py` (metadata), `theme.py`
(logo and CSS), `components.py` (pure display helpers) and `client.py` (one `call()` returning
ok / not_ready / failed). None of the four imports Streamlit, which is what makes them testable —
`app.py` still runs its page flow at import.

Open:
- `speech-emotion-recognition` has no trained checkpoint, so it answers 503 from `/ready`. Its
  pipeline, trainer and API are wired and verified.
- The eleven predecessor repos are archived on GitHub. `backup/pre-squash-2026-08-23` tags the
  pre-history-rewrite tip; delete it once the current history is settled.
