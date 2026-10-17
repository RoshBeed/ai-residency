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
`content-scoring/etl.py`, `word-embeddings/corpus.py` and
`vector-search-retrieval/cbow/train_tokenizer.py` all end in a word→id mapping, which looks
like the same duplication word2vec was. It is not. The first normalises real-world Unicode out
of Hacker News titles and keeps `gpt-4` and `c++` whole; the second lowercases text8, which is
already ASCII, and counts; the third does the same for a different consumer. The genuinely
common part is about twenty lines — a `Counter`, a sort by frequency — and extracting that
would buy less than the indirection costs. Two implementations of *the same algorithm* is a
library; two pipelines that happen to end in a dict is not.

What *did* become a library is scoring: `word2vec.benchmarks` holds WordSim-353, SimLex-999
and the Google analogy scorers, because two services measure embeddings now —
`word-embeddings` trains them and `content-scoring` extends the vocabulary onto a new corpus
and has to show it did not damage them doing so.

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

`word-embeddings` and `content-scoring` are the exceptions and are trained for real.

`word-embeddings` is trained on text8 and scored against WordSim-353, SimLex-999 and the Google
analogy set with published word2vec-on-text8 numbers beside them. Changing it means keeping those
numbers true: `make -C services/word-embeddings evaluate` reports them from a checkpoint, and the
README quotes them.

`content-scoring` is trained on 872,554 Hacker News posts, split by time, with four fusion
architectures compared against four baselines. `late` is built as `early` with the cross-group
weights deleted -- same input, width, depth and head -- so the two differ only in connectivity.

**Where the inputs meet does not affect the score here, and it took ~2,000 runs to say that
properly.** At matched hidden units early beats late only at the smallest budget tested (0.2526 vs
0.2288 at 128 units, p=0.045); from 256 up they are indistinguishable, early is flat across the
whole range and late climbs, because late starts throttled by its per-group blocks and stops being
throttled. The reason is the data. Fitting a model that *cannot* represent an interaction -- a tower per input
reduced to a score, then summed -- against one that can, over eleven pairings of the real target:
**every pairing is null or negative**, and three of the four that reach significance point the wrong
way, the joint model being worse. The task is additive, and no fusion architecture can beat another
when there is no interaction to model.

**Use the capacity-matched control.** A joint model has more parameters than a same-hidden-size
additive one, so it can win for reasons unrelated to interaction. The `title x all metadata` cell
read +0.0051 (p=0.021) against an unmatched control and +0.0016 (p=0.385) against an additive model
with towers three times wider. The earlier "+0.004 interaction" figure quoted in this file and the
README came from the unmatched comparison and did not survive.

**Four independent measurements now say the architectural distinction only matters when something
else is constrained** -- the fusion-point curve, the width sweep, the block-allocation ranking and
the robustness probe. Late fusion's one remaining advantage, degrading gracefully when a modality
goes missing, is itself a small-model effect: at 128 hidden units it pays 0.012 undamaged to insure
0.031, at 512 it pays 0.010 to insure 0.006 and the missing-metadata edge stops being significant
(p=0.24). Early fusion recovers with capacity where late fusion does not. Do not quote a fusion
result without the width it was measured at.

That null is only reportable because a synthetic control shows the same code detecting an
interaction the moment one exists: on `g + h + α·g·h`, Snoek late fusion falls behind by 0.118 at
α=8 and eight times the width recovers nine percent of that. The same control also found that
**early fusion and "towers into a shared trunk" are indistinguishable at every α** (max difference
0.0006), so this service's late-fusion models were never the kind the papers warn about. Earlier
readings of this comparison were confounded three times over -- by tower width, by a bare linear
head that made late fusion a generalised additive model, and by proportional block sizing that gave
the 300-d title 233 of 256 units and the author's three scalars 2. Do not quote a fusion number
without checking which of those was in force.

Same rule as word-embeddings: `make -C services/content-scoring evaluate` is the single source of
every number in its README, and the trainer streams the same harness to W&B rather than computing
its own metrics — so a chart and the table cannot disagree. The leave-one-group-out ablation runs on
the dense model, because the block-diagonal one splits its hidden units in proportion to input width
and an ablation there reports that allocation as much as the signal.

**Its ceiling is the representation, not the architecture, and that took four wrong turns to
establish.** A mean of static word vectors throws away word order; a fine-tuned MiniLM beats the
shipped model by +0.017 ρ and +19% on finding front-page posts, and a LoRA-adapted Qwen-0.5B beats
it on fewer than half the rows. Capacity over the *fixed* representation does nothing — depth,
boosting leaves and a boosted-tree ensemble all failed to help — which is a much narrower claim than
"capacity does not help" and was stated too broadly more than once before the encoder was tested.
The probes are in `notebooks/content-scoring/`; the write-up's §1.6 carries the numbers.
Its data path is credential-free end to end (`make etl && make extend-vocab && make train &&
make evaluate` on a clean machine), and the one published data artifact is the raw mirror at
`roshbeed/ai-residency-hn-posts`; splits and the title corpus are a local derived cache, never
uploaded.

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

**Read the error before forming a theory.** `word-embeddings` answered 503 from the published
image and the first explanation reached for was a slow checkpoint download, so the README's curl
retry budget got raised from 60 to 300 and then rewritten twice. None of it was the problem. The
container log said what the problem was, in one line, the whole time:

```
Pulled cbow_model.pt from roshbeed/ai-residency-word-embeddings
Loaded checkpoint from /app/services/word-embeddings/checkpoints/cbow/cbow_model.pt
ERROR  Could not load embeddings, /nearest will return 503: 'word_to_lemma_index'
```

The download worked and took four seconds. It was a `KeyError`: an image cached from an earlier
visit carried the vocabulary reader from before the word/lemma index spaces were untangled, while
the checkpoint on the Hub had since been re-uploaded with the new key. `docker run` reuses a local
`latest` and never re-checks it, which is why the quickstart now passes `--pull always`.

Two habits follow, and both were broken here.

**Get the output yourself.** `docker logs <container>` is one command. Asking someone else to run
it, waiting, and theorising in the meantime is slower than running it and is not debugging. If a
tool genuinely cannot be reached, say so once, plainly, and say what is blocked — do not keep
asking for output while continuing to guess.

**Do not test around a tool.** `uv` was missing on a laptop, so a shim stood in for `uv run` and
skipped `uv sync` entirely. Everything passed. CI then failed on `content-scoring`, because the
stale venv already had `types-tqdm` that the lockfile did not declare — exactly the bug the sync
would have exposed. Install the tool. A harness that supplies what the real one would have caught
reports green and means nothing.

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
  `--wait` blocks until every Deployment reports available, and a service with no
  checkpoint answers an honest permanent 503, so the rollout hits the progress
  deadline. Helm then marks the release `failed` even though all ten pods rolled out
  on the new tag. Every service has a checkpoint as of the speech-emotion push, so
  this is latent rather than active -- it returns the moment one does not. That is
  why `make helm-upgrade` does not pass `--wait`; don't add it back until every service
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
- **A newly pushed checkpoint repo has to be public, or serving cannot see it.**
  `push_checkpoint` defaulted to `private=True`, so the first push for a service created a
  private repo; the push logged success, and every container then failed with `Repository Not
  Found ... make sure you are authenticated`, which reads like a broken token rather than a
  visibility setting. Serving is credential-free by design, so private is never the right
  default here. It is `private=False` now. The existing repos were public already, which is why
  this only showed up on `speech-emotion-recognition`, the first new one in a long time.
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
- **Every figure is written at `ai_residency_common.figures.DPI`.** Nine services drew their own
  charts and the committed assets came out at 110, 140, 150, 200, 300 and 600, which a reader sees
  as figures that will not line up. Charts go through `figures.save_figure`; a screenshot or a slide
  export gets stamped by `python deploy/stamp_dpi.py --write`, which rewrites the `pHYs` chunk and
  touches no pixels. `deploy/test_workspace.py` fails on an asset at any other resolution.
- Validate every Mermaid block: `npx -y @mermaid-js/mermaid-cli -i b.mmd -o b.svg`. On a Mac the
  bundled Chromium fails to launch; point it at the installed browser instead:
  `echo '{"executablePath":"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}' > p.json`
  then add `-p p.json`. Extract the block with a fence-aware regex -- a `sed` range that stops at
  the first `\`\`\`` swallows the prose after it and reports a parse error in text that is not
  part of the diagram.

## State

**Nothing here is deployed, and the repo is public.** The write-up that is published is the
Quarto blog at roshbeed.com, which replaced the Streamlit frontend as the public face of these
projects; this repo is code on GitHub and its images are built but not served anywhere.

Going public also fixed CI. Every push-triggered run from 2026-08-29 failed in about five
seconds without starting a job -- "recent account payments have failed or your spending limit
needs to be increased" -- and the weekly `schedule` runs stayed green the whole time because
every job but the vulnerability scan is gated off `schedule`, so a green tick meant nothing had
run. Actions minutes are free on a public repo. Serving the
projects again is wanted later, so nothing in `deploy/` or `services/frontend` has been
removed — it is dormant, not deleted, and `ai-residency-frontend/docs/` holds the
superseded Streamlit write-ups.

Where a blog post measured something this repo had not, the service README records it and
says the number is the notebook's rather than a shipped checkpoint's — see
`vector-search-retrieval` §6 on in-batch negatives and `multimodal-captioning` §6 on how
much of the captioning gap more data actually closes.

CI is one path-filtered workflow: lint, then per-member mypy and pytest, then build → smoke-test →
publish (only the images that changed) gated behind a `PUBLISH_IMAGES` repository variable, then a
Trivy scan of the published base whose results go to the job summary (SARIF upload needs Advanced
Security, which a private repo does not have). 380 tests.

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

## Getting every service to a meaningful checkpoint

The current pass: each service should show that its mechanism works, not that it is
state of the art. Results need to beat a stated baseline and be reproducible from the
Hub, cheaply -- not perfect, and not a long training run. Order of work per service:

1. Find the quickest change that makes the result meaningful. Measure the obvious lever
   before spending hours on it: for captioning, Section 6 had already shown 16x the data
   buys almost nothing, and the real bottleneck was a single pooled image token.
2. **Resumable training first.** This Mac dies when it runs out of RAM, so a run that
   cannot resume will eventually lose a night's work. Use `resume_from`, `save_resume`
   and `resume_path_for` from `ai_residency_common` -- one implementation, tested in
   `libs/common/tests/test_resume.py`, not re-written per trainer. A trainer needs three
   lines: derive the path, declare a `fingerprint` of what makes this the same run, and
   call `save_resume` every epoch rather than only on an improvement. `default_best` is
   `0.0` for accuracy-ranked trainers and `inf` for loss-ranked ones.
3. Train, with the baseline logged beside the metric every epoch.
4. Push to the Hub on best. Serving pulls credential-free, so the repo must be public --
   see the `push_checkpoint` note above.
5. Verify through the container, not the local venv: `up -d --wait` and a real request.
6. Then update every place the old result is written down.

Step 6 is the one that gets forgotten. The places a result appears:

- the service's own `README.md` -- its results section, and any figure showing captions,
  predictions or scores
- the root `README.md` -- the systems table and the resident-memory table
- `CLAUDE.md` -- the standing notes and the Open list below
- the frontend, where a page describes what the model does
- the published image, which CI rebuilds on push -- so the service only serves the new
  checkpoint once that lands

Status:

- [x] `word-embeddings` -- trained on text8, scored against WordSim-353/SimLex-999/Google
      analogies with published numbers beside them.
- [x] `content-scoring` -- trained on 872,554 Hacker News posts, four fusion architectures
      against four baselines.
- [x] `speech-emotion-recognition` -- whisper-tiny on English CREMA-D, val_acc 0.7225
      against a 0.1708 baseline. Was one epoch of German EmoDB at below chance.
- [x] `speech-emotion-recognition` -- Section 6 rewritten with the real numbers; root README's
      resident-memory row now reads 0.9 GB and 0.7225 vs 0.1708. Verified through the container
      before updating either.
- [x] `speech-emotion-recognition` -- resume support, on the shared mechanism.
- [ ] `speech-emotion-recognition` -- CREMA-D comes from `myleslinder/crema-d`, a
      third-party account. Mirror it under `roshbeed/` like text8, and create the
      `-data` repo, which does not exist.
- [ ] `multimodal-captioning` -- 49 patch tokens instead of one pooled token; training on
      5,000 Flickr30k pairs. Then: score with CLIP agreement against the published 0.1882
      untrained / 0.3280 human numbers, rewrite Section 3.2, Figure 5 and Section 6,
      and decide whether `adapter_patches.pth` stays the default.

      **The write-up must not claim the patch grid caused the improvement.** Section 6's
      published loss numbers (6.4251 baseline, 6.0981 best) come from 100 pairs over 3
      epochs; this run is 5,000 pairs over 12. Three things changed at once -- image tokens
      1 to 49, data 50x, epochs 4x -- so the loss comparison attributes nothing. It was
      presented as an isolated measurement of the patch grid once, and it is not one.
      Isolating it would mean training the single-token path at the same 5,000 pairs and 12
      epochs, roughly four more hours, for an attribution this phase does not need. What
      the result has to clear is CLIP agreement against 0.1882 untrained and 0.3280 human.
- [ ] `multimodal-captioning` -- then measure, not extrapolate, what SmolLM2-135M buys as
      the decoder. The 2.9x estimate is arithmetic, and tonight's arithmetic was wrong by
      ~40x in both directions.
- [ ] `vit-classifier`, `vit-seq2seq`, `whisper-fine-tuning`, `rlhf-ppo`,
      `vector-search-retrieval` -- not yet assessed in this pass. For each: what does it
      currently answer, what is its baseline, and what is the cheapest change that makes
      the answer meaningful.

### Backbone sizing, researched 2026-10-17

Smaller backbones were strictly better for `speech-emotion-recognition`: whisper-small to
whisper-tiny went from below-chance to 4.2x baseline *and* from 2h50m to 4 minutes an epoch.
So the question got asked of every service. Most are already minimal:

| service | backbone | params | verdict |
|---|---|--:|---|
| `speech-emotion-recognition` | whisper-tiny | 39M | done |
| `whisper-fine-tuning` | whisper-tiny | 39M | already minimal |
| `rlhf-ppo` policy | SmolLM2-135M-Instruct | 135M | already minimal |
| `rlhf-ppo` reward | deberta-v3-large-v2 | 435M | inference only; see below |
| `multimodal-captioning` | CLIP ViT-B/32 + Qwen3-0.6B | 151M + 596M | decoder is the one real candidate |

**Swap the captioning decoder for `HuggingFaceTB/SmolLM2-135M-Instruct`, not the vision
encoder.** SmolVLM-256M -- the smallest working VLM -- is SmolLM2-135M plus SigLIP, so a
135M decoder is proven for exactly this job. 596M to 135M params, but the honest figure is
compute: layers x hidden^2 is 28x1024^2 = 29.4M against 30x576^2 = 10.0M, so **2.9x, not
4.4x**. SmolLM2 is *deeper* (30 layers vs 28) and much narrower, and most of Qwen3-0.6B's
parameter advantage is a 151k-token embedding table that costs almost nothing per token.
The adapter's output width drops 1024 to 576 with it, which invalidates every existing
adapter checkpoint.

**Do not swap CLIP for SigLIP, tempting as the better features are.** It is the wrong
direction twice: `siglip-base-patch16-224` is 203M and `siglip2-base` 375M against CLIP
ViT-B/32's 151M, and patch16 at 224px yields **196 image tokens against CLIP patch32's
49**. Training cost here is the backward pass through the frozen decoder over the image
sequence, so 4x the image tokens would more than cancel a 2.9x smaller decoder. CLIP also
does double duty: Part 1 of that service is CLIP zero-shot classification at 10/10.

**The reward model has a same-recipe smaller sibling, `OpenAssistant/reward-model-deberta-v3-base`**
-- compute 7.1M against 25.2M, 3.5x cheaper, and still "a real preference-trained reward
model". Two reasons to measure before taking it: it is inference-only inside PPO, so it may
not be the bottleneck at all, and reward quality shapes what PPO learns, so a weaker reward
model means a worse-behaved demo. 10,803 downloads against 2,038 says large-v2 is the
better-validated one.

**Measure, do not extrapolate.** Tonight's estimates were wrong by more than an order of
magnitude in both directions: whisper-small to tiny was predicted at 12x and measured at
~500x per step, and "one epoch needs hours" turned into 4 minutes. There is a timing probe
pattern that works -- build the real model and dataloader, time 8 steps after 3 warmup
steps, at a couple of batch sizes -- and it costs minutes.


Open:
- `speech-emotion-recognition` is trained for real now: whisper-tiny on English CREMA-D,
  **0.7225 held-out accuracy against a 0.1708 majority-class baseline**, 4.2x chance over six
  emotions, and 20/30 held-out clips correct through the container. Held-out loss bottoms at
  epoch 6 and accuracy peaks at 10, with training loss down to 0.0178 by then, so eight to ten
  epochs is the useful range rather than twenty.

  Getting there corrected two earlier mistakes worth keeping written down. The first attempt was
  one epoch of German EmoDB scoring 0.2342 against a 0.2382 baseline -- *below* chance -- pushed
  to make the page demonstrable rather than because it worked.

  That first run took 2h50m for one epoch, which got blamed on the laptop and should not have
  been. **Whisper's feature extractor pads every clip to 30 seconds** -- 3000 mel frames -- and
  EmoDB clips average 2.5s, so 92% of every forward and backward pass was silence. Combined with
  a full fine-tune of all 281.6M parameters of whisper-small, on MPS where some ops fall back to
  CPU. whisper-tiny is a quarter the width and a third the depth, so roughly a twelfth of the
  encoder cost; the same run is minutes, not hours. The point of this service is the mechanism --
  emotion as a token the decoder emits -- not the size of the backbone, so the backbone should be
  the smallest one that shows it.
- `content-scoring` ships a `log_mse` objective, which ranks well and predicts about a fifth of the
  upvotes that actually happen. `poisson` is implemented and predicts the count directly; a binary
  head on `score >= 100` finds front-page posts best and cannot rank at all. Which of the three is
  right is a product question the service has not been asked.
- The eleven predecessor repos are archived on GitHub. The four `backup/*` tags that held the
  pre-rewrite tips are deleted from the remote now the repo is public; they survive locally only.
