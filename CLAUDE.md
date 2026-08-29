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
services/<name>/             one per system: package, config/, serving/, tests/, README
services/_template/          the shape a new service starts from; a working one, not a skeleton
services/frontend/           the Streamlit app, one module per system under views/
Dockerfile.{base,serve,train}  every image, at the root, selected by --build-arg SERVICE=
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

**One package per service, not nine.** `rlhf-ppo` had `advantage`, `base_setup`,
`experience_buffer`, `models`, `ppo_trainer`, `reward_kl`, `sa_pair`, `token_attribution` and
`training` as top-level packages -- two of them holding a single type alias and a single class.
The cost was not the file count but the namespace: mypy already has to run per service because
nine services each define a top-level `config`, and generic names like `models` and `training`
made that worse. It is one `rlhf_ppo` package now, laid out by PPO stage. The eight other
services already did this.

**Vocabulary building is deliberately not shared, and that was checked rather than assumed.**
`word-embeddings/etl.py` (947 lines) and `vector-search-retrieval/cbow/train_tokenizer.py`
(117) both end in a word→id mapping, which looks like the same duplication word2vec was. It
is not. The first normalises real-world Unicode, pulls from a Hacker News Postgres mirror and
a Wikipedia dump, merges corpora and lemmatises; the second lowercases text8, which is already
ASCII, and counts. The genuinely common part is about twenty lines — a `Counter`, a sort, and
a reserved `<PAD>` at id 0 — and extracting that would buy less than the indirection costs.
Two implementations of *the same algorithm* is a library; two pipelines that happen to end in
a dict is not.

**CI's path filter is `dorny/paths-filter`, and its filter list is checked against the
workspace.** The diff was the part that kept being subtly wrong: resolving a base SHA
needs one rule for `pull_request` and another for `push`, and the hand-rolled version
had already produced two bugs -- an empty diff killing the step under `set -e`, and an
empty selection serialising as `[""]`, a matrix with one nameless entry. The filters are
listed rather than derived from the directory tree, which is a real trade: a list can go
stale, and it goes stale silently, because a member with no filter is simply never
selected and CI stays green while running none of it. `deploy/test_workspace.py` is what
makes that safe -- it fails if the filters and the workspace disagree in either
direction.

**The manifest is consumed, never propagated.** `deploy/images.yaml` is a Helm values
file: `helm upgrade -f ../images.yaml` merges it over the chart's defaults, so
`values.yaml` holds no digests and cannot disagree with it. Compose reads
`compose/images.env`, written from the same manifest, through
`image: <repo>${<SERVICE>_REF:-:latest}` -- and an unset variable falls back to
`:latest`, which is what `make local-up` gets by passing `--env-file /dev/null`. This
replaced a script that rewrote `values.yaml` and `docker-compose.yml` with regular
expressions, plus a third generated file whose only job was undoing the pinning. Drift
used to be *detected*; now there is one generated file and nothing else to disagree.
`base-image.digest` is separate on purpose -- it is a build input, not something the
chart deploys.

**A container image declares how it runs; nothing downstream restates it.** Training
Jobs and compose's training services both use the image's own `CMD`, which comes from
`[tool.ai-residency] train-module`. The chart used to carry a `trainCommand` per
backend, and it went stale the moment rlhf-ppo's packages moved -- `make helm-train
SERVICE=rlhf-ppo` launched a Job that died on `ModuleNotFoundError` while the image was
correct all along. Compose never had the bug, because it never restated the command.

**Compose repeats itself through `x-` anchors, not by copy-paste.** The readiness
healthcheck was written out nine times and the credential pair eighteen -- ~90 lines of
a 367-line file, and nine copies of one contract that could drift a service at a time.
Compose's own extension point holds them: a top-level `x-backend-healthcheck` /
`x-hub-credentials` that Compose ignores, aliased with `*name`, merged with `<<:` where
a service needs extra variables alongside. Verified by diffing `docker compose config`
before and after: all twenty services resolve identically.

**The chart derives what it can from the backend key.** Image repositories are
`<registry>/<chart>-<service>-serve` and `-train`, built in `ai-residency.repository`
from the `range` key rather than written out eighteen times in `values.yaml` beside the
very key they derive from. `containerPort` and `cpuRequest` are chart-level defaults in
`backendDefaults`, having been one distinct value across nine backends. The measured
`memoryRequest` / `memoryLimit` stay per service -- those genuinely differ. An explicit
`image:` still wins for a service that breaks the convention; `deploy/test_workspace.py`
fails on one that spells out what the chart would derive anyway.

**`HF_HOME` is set once, in `Dockerfile.base`, and nothing downstream repeats it.** The
base sets it so the path the chart mounts and the path huggingface_hub writes cannot
drift apart, and then compose overrode it to `/root/.cache/huggingface` -- unwritable by
UID 1000 -- while the chart restated the correct value for two backends. Which services
mount the cache is now a list on `hfCache`, with the mountPath given once and asserted
against the Dockerfile.

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

**Structure first; model quality later -- except where a service has moved past it.** The default
phase is that everything is plumbed correctly end to end: untrained or weakly-trained checkpoints
are expected, not a defect, and a service that starts, reports `checkpoint_loaded: false` and
answers 503 honestly is *passing*. Verify wiring, not accuracy, and don't propose training runs as
the next step.

`word-embeddings` is the exception and is now trained for real, on text8, scored against
WordSim-353, SimLex-999 and the Google analogy set with published word2vec-on-text8 numbers beside
them. Changing it means keeping those numbers true: `make -C services/word-embeddings evaluate`
reports them from a checkpoint, and the README quotes them.

The lesson generalises even for the services still in the default phase. Four separate defects in
that trainer were invisible to every structural check -- the tests passed, the API answered, the
loss went down. What exposed them was one number: the loss a *random-init* model reports. The
shipped checkpoint sat at 10.02 against a random baseline of 11.09, which reads like a trained
model until you know the untrained value. Log the baseline next to the metric, always.

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

**Images are published for amd64 *and* arm64, and dropping arm64 is a trap.** It is
QEMU-emulated on an x86 runner and measurably expensive — 8.4 minutes across a full
eleven-image publish — which is why it was dropped once. That number stopped applying
the moment only changed images rebuilt: a typical commit builds one image, so arm64
costs about fifty seconds. What amd64-only cost instead: `docker run` **fails** on Apple
Silicon rather than falling back to emulation (`no matching manifest for
linux/arm64/v8`), so the README's first command broke for most of its readers, and
Docker Desktop's Kubernetes node — also arm64 — could not pull any image at all.

**A local cluster can run locally-built images.** `make -C deploy local-cluster` builds
natively, imports into the node's own store (a host `docker build` is invisible to it),
and installs *without* the `images.yaml` overlay, so the chart falls back to
`image.tag` -- a locally-built image has its own digest and would never match the
manifest's. Skaffold was considered for this loop and turned down: the load step cannot
be delegated (Docker Desktop does not expose its node through the kind CLI), and
published images have been multi-arch since the arm64 restore, so this is an escape
hatch rather than the normal path. Not required now that published images are
multi-arch — useful for iterating on something unpublished.

**One Dockerfile per image kind, at the root, parameterised by `SERVICE`.** There were
ten `Dockerfile.serve` and ten `Dockerfile.train`; six of each were byte-identical once
the service name was substituted, and the rest differed by a redundant `COPY` or a
comment. That is not ordinary copy-paste: what was duplicated twenty times is the `uv
sync --no-install-package torch ... --inexact` line documented below as subtle, so
correcting it was a twenty-site edit. `make build SERVICE=x` and CI both pass
`--build-arg SERVICE=`. `services/frontend` keeps its own `Dockerfile.serve` -- it does
not build FROM the torch base and does not run a FastAPI app. A test fails if a
per-service Dockerfile reappears.

**Images install with `--no-dev`.** uv installs the `dev` dependency-group unless told
not to, so every serving image shipped mypy, pytest, ruff and httpx -- a published
backend carrying a type checker and a linter it can never run, 150MB on
word-embeddings and 134MB on the frontend. It is also the split that keeps a service's
dev-only dependencies out of a runtime that never imports them. Runtime extras a
training stage genuinely needs go in `[project.optional-dependencies]`, which
`Dockerfile.train` installs with `--all-extras` and the serving image does not install
at all.

**The training entrypoint is declared in the service's own pyproject.toml.**
`[tool.ai-residency] train-module` is the one thing that differs between training
images, so it lives next to the code it names rather than in a table in the Makefile or
in CI. Both read it; `deploy/test_workspace.py` fails if a service omits it or names a
module that does not exist.

**Only changed images are built, and nothing is retagged.** CI's publish matrix
covers `changes.outputs.services`, so an unchanged image gets no build and no tag it
did not earn. This replaced a scheme that rebuilt all eleven every push -- 78% of this
repo's Actions minutes -- and then, briefly, one that retagged the unchanged ones. Both
existed only because `deploy/` pinned a single tag across every image; the manifest
removed the reason for either.

**Images are published to Docker Hub publicly** (the free tier allows one private repository, and
there are twelve images). CI pushes the serving base and the images that changed on every push to
`main`, gated behind the `PUBLISH_IMAGES` repository variable and the `DOCKERHUB_TOKEN` secret.
Publishing follows the same path filter as the checks -- see the rule above on why the manifest
removed the reason to ignore it.

Training images go through the separate `Publish training images` workflow, run on demand: they are
~9GB each, and the base plus one service does not fit in a runner's default free disk, so that
workflow clears the preinstalled toolchains and builds one service per job. Runners are x86-64, so
those builds are native rather than the slow emulation a Mac would do.

**Training images carry their own tag, and must not share the serving one.** `deploy/images.yaml`
covers serving images only -- nothing resolves a digest for a ~9GB image published on demand by a
different workflow. They were rendered as `{{ .Values.image.tag }}`, which the manifest stopped
moving, so `make helm-train` was pinned to whichever commit last ran the old `make pin-images` and
would have stayed there forever. The chart uses `train.imageTag` and compose uses
`TRAIN_IMAGE_TAG`, both defaulting to `latest` -- the tag that workflow pushes alongside its sha-
tag. Pin a run with `--set train.imageTag=sha-<commit>`.

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
- **An embedding matrix can have more rows than the vocabulary has words.** The old
  word-embeddings checkpoint was 258,196 rows against 230,701 words, so 27,495 rows
  could not be named. `topk` searches all of them, a nameless row wins, and the lookup
  after it raises `KeyError` -- `/nearest` returned 500 for any `k` above about 9
  while the schema advertised up to 100. Never index a vocabulary with a raw `topk`
  result; go through `word2vec.nearest_words`, which masks the unnamable rows out of
  the search.

  The *cause* was worse than the symptom, and is fixed now: the corpus was encoded with
  `word_to_index` (word ids) while the checkpoint saved `word_to_lemma_index` (lemma ids), so
  serving and `content-scoring` were both reading a matrix through a mapping that indexed a
  different space. Every neighbour it ever returned was junk. A checkpoint carries
  `word_to_index` now, and the readers prefer it and fall back to the old key with a warning.
- **Analogy search must exclude all three query words.** `king - man + woman` lands nearest to
  `king` itself far more often than to `queen`, so a scorer that excludes none of them reports a
  near-constant zero however good the vectors are -- which is exactly what the old
  `evaluate_embeddings` did, and why a real defect hid behind a metric for months.
  `word2vec.nearest_words` takes `exclude` as a word *or an iterable*; pass all three.
- **A W&B sweep passes nothing without `${args_no_hyphens}`.** `${env}` alone expands to
  `/usr/bin/env` and no parameters, so every run trains the defaults, reports a near-identical
  score, and the Bayes search optimises noise for as long as you let it. The pair must also come
  *before* the program in `command:`, which is what makes them environment variables rather than
  argv. The previous `sweep.yaml` had `${env}` alone plus a comment claiming the agent exported
  the parameters.
- **Bound a long sweep by wall clock, not by `--count`.** Run times in one sweep span 6 to 31
  minutes depending on which hyperparameters a run drew, so a count chosen for the average
  finishes anywhere between half and double the budget. `make sweep SWEEP=<id> HOURS=12` wraps the
  agent in `caffeinate -is timeout 12h`; stopping early costs nothing, since finished runs are
  recorded and the next agent reads them back.
- **A credential in the environment is not the only credential.** `get_hf_token` checked
  `HF_TOKEN`/`HUGGINGFACE_TOKEN` only and `init_tracking` checked `WANDB_API_KEY` only, so on a
  machine logged in through `huggingface-cli login` or `wandb login` a run trained for hours, then
  warned once that it was keeping the checkpoint local, and logged to an offline directory nobody
  looked at. Both now fall back to the credential store the CLIs write.
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
- **`Config` fields typed as `Literal` need `_coerce` to know about `Literal`**, which it now
  does -- and it validates the value against the allowed set while it is there. Before that, a
  `Literal` field silently could not be set from the environment, which excluded it from every
  sweep.
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

CI is one path-filtered workflow: lint, then per-member mypy and pytest, then build → smoke-test →
publish (only the images that changed) gated behind a `PUBLISH_IMAGES` repository variable, then a
Trivy scan of the published base whose results go to the job summary (SARIF upload needs Advanced
Security, which a private repo does not have). 330 tests.

Building images locally is for debugging, not delivery — CI is what publishes. `make base` and
`make build` exist so a change to a Dockerfile can be exercised before it is pushed.

`services/frontend` routes with `st.navigation`: `app.py` is 85 lines that map a service name to
its module in `views/`, and each page is an ordinary `render()`. That replaced an 892-line
`if _PAGE == ...` chain at module scope plus a session-state and query-param dance to keep the URL
and the sidebar agreeing. Streamlit owns the URL and history now, and because a page is a function
rather than a script body, `AppTest` renders all nine headlessly — both the ready and the not-ready
path — which took the frontend from 33 tests to 53. `shared.py` holds the framing every page shares
(`service_page`, the readiness gate); `projects.py`, `theme.py`, `components.py` and `client.py` are
unchanged and still Streamlit-free.

`_template` is checked but never published. It was excluded from both CI matrices, so the seed for
every new service was linted, typed and tested on a laptop and nowhere else; the publish matrix
still drops it, since nothing needs to ship a template image.

Open:
- `speech-emotion-recognition` has no trained checkpoint, so it answers 503 from `/ready`. Its
  pipeline, trainer and API are wired and verified.
- The `base-serve` digest in `deploy/base-image.digest` is single-platform (amd64). Any push touching a
  shared input rebuilds the base multi-arch and the `lock` job replaces it, so it self-heals; the
  `images` job now fails with a named error rather than "no match for platform in manifest" if a
  service-only push ever meets a single-platform base.
- The eleven predecessor repos are archived on GitHub. `backup/pre-squash-2026-08-23` tags the
  pre-history-rewrite tip; delete it once the current history is settled.
