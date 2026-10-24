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

The same rule covers a corpus that is nominally reachable but not *practically* so. CREMA-D's
audio sat behind a multi-gigabyte `git lfs clone` of a repo carrying video and perceptual ratings
this project never reads, so the documented `download-data` had never once been run and the
shipped checkpoint was not reproducible from the shipped pipeline. The audio half is mirrored at
`roshbeed/ai-residency-speech-emotion-recognition-data` under CREMA-D's own ODbL, verified
byte-identical to the clips that trained the checkpoint. When you mirror a corpus: check the
licence, put it and the citation on the dataset card, and verify the copy against the original
rather than assuming the download was faithful.

**The licence decides it, not the size.** text8 is public domain and CREMA-D is ODbL, which is
what made mirroring both of those right. Flickr30k is not mirrored and will not be: its authors
state they do not hold copyright in the images, use falls under Flickr's own Terms of Use, and
the data is offered at their link for non-commercial research with no redistribution right. A
corpus nobody may re-host leaves a training path that depends on someone else's upload, and that
is a property of the corpus to disclose rather than a gap to close. What must not depend on
another account is *serving*, and nothing does.

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

**One preprocessing path, or a test that the two agree.** Training builds its inputs in the
dataset; serving builds them from whatever the caller posted. Where those are written twice they
drift, and the drift is invisible: `vit-seq2seq` normalised its MNIST digits and then normalised
the finished composite again, so training strokes arrived at ~3x the amplitude the API ever
produces and the published accuracy was **97.93% on the loader's tensors against 64.23% on the
API's**, measured on the same images. Share the function where one exists -- captioning uses
`backbones.image_processor` on both paths, the speech services `load_model_and_processor`. Where
it is hand-rolled, assert the round trip: recover pixels from a training tensor, encode the file a
caller would send, push it through the service's own preprocessing and compare. Scoring a
checkpoint on inputs no request can produce is the same error as ranking it on a loss from a
different objective.

**Augment the layout, not just the content -- and the frontend is part of the serving
path.** Third instance, and the worst. `vit-seq2seq` composited MNIST digits at a fixed
28px cell into evenly spaced slots in a centred strip, then applied rotation, scale, shear
and translation *inside* that cell. The handwriting varied across 270,000 examples and the
geometry never did, so the model learned the layout: 92-100% exact-sequence for digits
between 22 and 47 pixels, **3.3% at 20 and 10.0% at 95**. The frontend offers a 280x280
canvas and a 14px brush, so a person drawing one digit fills it -- about 95 pixels once
scaled. Every number in the write-up was measured on composites built by the training
generator, including the ones called "verified through the container", because the PNGs
posted to the API came from that same generator. The user found it by drawing ten numbers
and getting ten wrong answers.

Two habits follow. **Generate the training data the way the input is actually produced**,
which here meant drawing the cell size, the gaps and the group's position per composite
rather than warping glyphs inside a frozen grid. And **when a service has a UI, the UI is
the serving path** -- `make build` plus a `curl` with your own fixture is not an
end-to-end check, it is the same fixture twice. Report the metric split by whatever the
real input varies in (`make evaluate` prints accuracy by ink height now), so a collapse at
one end cannot average away.

It happened twice in one pass, which is why this is a rule rather than a note.
`vector-search-retrieval` was the second: the offline tokenizer that filled the Redis index
lowercased and `retrieval/query_engine.py:tokenize` did not, so `/query` scored **0.0020 recall@10
on title-cased queries -- the chance floor** -- while the evaluation reported 0.6728 on the
all-lowercase queries MS MARCO happens to ship. **Test the input the evaluation cannot produce.**
Held-out data is drawn from the training distribution by construction, so the one case it can never
cover is the one a real caller supplies: a capital letter, a PNG round trip, a trailing comma.

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
- **An unqualified Hub dataset name no longer resolves.** `load_dataset("ms_marco")` had
  worked for as long as the Hub accepted bare names; current `huggingface_hub` refuses with
  `Repository id must be 'namespace/name'`, which killed every `vector-search-retrieval`
  stage that reads the corpus. It is `microsoft/ms_marco` now, named once in `Config` rather
  than in each of three modules. Any other bare name in this repo is a latent version of the
  same break.
- **`tar -t` on macOS hides what Python's `tarfile` sees.** A tarball built with bsdtar from
  a directory with extended attributes carries one `._name` AppleDouble member per file, and
  `tar -tzf` does not list them -- it folds them back into metadata -- so a listing looks
  clean while `tarfile` iterates twice the members. The first CREMA-D mirror shipped 7,442 of
  them. Build published archives with `COPYFILE_DISABLE=1 tar --exclude='._*'`, and count
  members with Python rather than with `tar -t`.
- **`.python-version` pins 3.11.** Without it uv picks per-host and mypy rejects numpy's stubs.
- **buildx resolves a base-image tag against the registry, not the local store.** `make base`
  followed by `make build SERVICE=x` silently built against the *published* base until the Makefile
  started passing `--build-arg BASE_IMAGE=` explicitly. Same class of trap as the kind node's
  separate image store, one layer up.
- **`AI_RESIDENCY_<FIELD>` overrides any `Config` field** — `AI_RESIDENCY_NUM_EPOCHS=1` for a smoke
  run. `vit-classifier` uses `classifier_epochs`, so it needs `AI_RESIDENCY_CLASSIFIER_EPOCHS`.
- **Docker Desktop's VM goes read-only when the host disk fills**, and stays read-only until the VM
  itself restarts — quitting the app is not enough unless `com.docker.virtualization` actually exits.
- **Homebrew's `node` is currently broken on this machine, which blocks Mermaid validation.**
  `node` aborts before running anything: `Library not loaded:
  /opt/homebrew/opt/simdutf/lib/libsimdutf.36.dylib` (and the same for `libsimdjson.33`). The
  Cellar has simdutf 37, so a dependency upgrade moved out from under the installed node
  binary. `npx @mermaid-js/mermaid-cli` therefore cannot run at all, and the writing
  convention above depends on it -- so do not add a Mermaid block while this is broken, because
  it cannot be validated. `brew reinstall node` is the fix; it is a toolchain change rather than
  a repo one.

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
- **A figure nobody can redraw is a figure that goes stale.** Both ViT figures were slide
  exports and both drifted away from the model with nothing noticing: Figure 1 showed a
  four-digit composite laid out two-by-two, and the architecture figure claimed 16 patch
  tokens, `nn.Linear(11, 13)` and a `<pad>` id the model does not have. The prose had started
  apologising for them in line rather than fixing them. They come out of
  `vit_seq2seq/report.py` now -- `make -C services/vit-seq2seq figures` -- with every shape
  read off `Seq2SeqArch` and a real forward pass, and every decoded sequence a real `generate`
  call. `content-scoring` and `word-embeddings` already had a `report.py`; that is the pattern.
  A figure shared by two services is written by one run to both paths, so they cannot disagree.
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
Security, which a private repo does not have). 425 tests.

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
Hub, cheaply -- not perfect, and not a long training run.

**The architecture is the deliverable; its scale is not.** Hyperparameters, backbone
sizes and corpus sizes are all free to change in whatever direction makes a run cheap,
as long as the mechanism the service demonstrates survives intact -- a smaller
pretrained model, a slice of the data, a larger batch, anything that trains more
efficiently. whisper-small to whisper-tiny was the model for this: the mechanism was
untouched and the result went from below-chance to 4.2x baseline. What is *not* free is
the mechanism itself: late fusion stays late fusion, attention stays written out, the
adapter stays the only thing that trains.

**Train natively, never in a container.** Docker on this Mac has no GPU access, so a
containerized run lands on CPU and gives up MPS entirely. Containers are for *serving*
and for verifying a checkpoint through the deployed path. Training is `uv run` on the
host.

**MPS is the default, not always the right device, and `AI_RESIDENCY_DEVICE` overrides
it.** `rlhf-ppo` is the counterexample: a PPO step holds activations for four model roles
across twelve stages, and on MPS that accumulates past the backend's high-watermark
ceiling and dies mid-step with `MPS backend out of memory (MPS allocated: 42.45 GiB, max
allowed: 42.43 GiB)` -- after 252-413s in the value update alone. The same step on CPU
finishes in **70s total** with every stage finite, so CPU is both the faster and the only
working device there, and `make -C services/rlhf-ppo train-loop` pins it. Two wrong
theories came before that measurement: that the waste was the causal LM building
full-vocabulary logits to read one position (real, worth fixing, 143s to 103s on one
stage, did not stop the OOM), and that the batch size was too large (a 200x cliff between
16 and 32 on MPS, which was the ceiling showing up rather than the cause). Time the
stages before shrinking anything -- `ppo_step` takes an `on_stage` callback that makes
this one run.

Order of work per service:

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
3. Train, with the baseline logged beside the metric every epoch. Epoch counts are an
   upper bound, not a plan: `BestMetric` from `ai_residency_common` ranks epochs and
   stops after `patience` without a new best. Both guesses this pass were poor --
   speech-emotion ran 20 epochs where 9 would have done, captioning 12 where the
   curve turned at 3.
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
- [x] `speech-emotion-recognition` -- whisper-tiny on English CREMA-D, val_acc 0.7064
      against a 0.1708 baseline. Was one epoch of German EmoDB at below chance.

      **A corpus holds the recording constant; an upload does not.** CREMA-D is studio
      audio at one level with a clean floor, and the model had learned to rely on both
      rather than ignore them. Same harness, same 1,492 held-out clips, through the
      serving path -- clean-trained checkpoint against the shipped one:

      | upload differs by | clean-trained | shipped |
      |---|--:|--:|
      | nothing | 68.9% | 70.6% |
      | a sixteenth of the volume | 68.9% | 70.6% |
      | room noise, 20dB SNR | 58.2% | 71.0% |
      | room noise, 10dB SNR | 42.6% | 66.1% |

      Two fixes, because they are two problems. **Level is preprocessing**:
      `speech.load_waveform` scales to a fixed RMS, in the shared library because both
      services and all four of their training and serving paths go through it. Whisper's
      feature extractor normalises log-mel against each clip's own maximum, which does
      not rescue audio near the floor -- a sixteenth-volume clip read 25.3% against a
      16.7% chance line. **Noise has to be learned**, since scaling a noisy clip scales
      its noise: training adds white noise at 8-35dB SNR, train split only.

      Costs 1.6 points on clean studio audio (0.7225 under its own preprocessing to
      0.7064) to gain 23.5 at 10dB. And **changing the loader invalidated the previous
      checkpoint** -- fed normalised audio it never trained on it reads 0.689, which is
      why this was a retrain and not a patch.

      Two measurement mistakes worth not repeating. A first probe reported 13.3% on
      unperturbed clips -- `load_model_and_processor` without a checkpoint path returns
      base Whisper with freshly initialised emotion embeddings, which is chance for six
      classes. It was caught only because it contradicted a number already trusted.
      And a 150-clip subset said normalisation *hurt* noise robustness; on the full set
      it helps substantially (30.0% to 58.2% at 20dB). **A subset that disagrees with
      the full set is not a faster measurement, it is a wrong one.**
- [x] `speech-emotion-recognition` -- Section 6 rewritten with the real numbers; root README's
      resident-memory row reads 0.9 GB and 0.7064 vs 0.1708. Verified through the container
      before updating either.
- [x] `speech-emotion-recognition` -- resume support, on the shared mechanism.
- [x] `speech-emotion-recognition` -- CREMA-D's audio is mirrored at
      `roshbeed/ai-residency-speech-emotion-recognition-data` (7,441 clips, one 473MB
      tarball, ~20s), and `download-data` fetches it instead of a multi-gigabyte git-lfs
      clone that had never actually been run. Verified byte-identical to the clips the
      0.7225 checkpoint was trained on. ODbL, so the card carries the licence, the citation
      and a pointer to the authors' form. (An earlier note here said the data came from the
      third-party `myleslinder/crema-d`; that was a scratchpad script of mine, never the
      repo -- the repo always cloned the authors' own GitHub.)
- [x] `multimodal-captioning` -- 49 patch tokens instead of one pooled token, scaled to the
      decoder's embedding std, trained on 5,000 Flickr30k pairs for 15 epochs. **CLIP
      agreement 0.2115 against a 0.1848 untrained floor and a 0.3233 human ceiling: 19.2%
      of the gap**, 87/100 captions distinct. Section 3.2, Figure 5 (now Mermaid; the old
      PNG showed one token and an output of "some random gibberish") and Section 6 rewritten.
      `adapter_patches.pth` stays the default.

      **The write-up does not claim the patch grid caused it, and must not start.** Against
      the 100-pair checkpoint three things changed at once -- image tokens 1 to 49, pairs
      40x, epochs 5x -- so neither the loss nor the agreement comparison attributes anything
      to the grid. Isolating it would mean training the single-token path at 4,000 pairs and
      15 epochs, an attribution this phase does not need.

      Two things worth not relearning. **Loss is the wrong scoreboard for this service**: the
      pre-fix run cut held-out loss to 2.2043 from a 5.1933 baseline while scoring *below* an
      untrained adapter on agreement, because a frozen decoder lowers loss by learning caption
      register. **A soft token has to arrive at the scale of a real embedding**: the adapter
      ends in a LayerNorm (unit variance) and Qwen3's embedding std is 0.0302, so unscaled it
      was 33x too large and the cheapest way to cut loss was to stop varying with the image --
      per-image spread halved, to 0.104 against an untrained 0.227.

      Measured and rejected: `no_repeat_ngram_size=3` at decode time, 0.2270 against 0.2266.
      Decoding stays greedy.
- [x] Dataset sources, checked 2026-10-17: every `load_dataset` id in the workspace is
      namespace-qualified now, and `microsoft/ms_marco`, `CarperAI/openai_summarize_tldr`
      are org-hosted. The one that is not is `nlphuji/flickr30k`, a community re-upload
      standing in the same place a teammate's text8 repo once did.

      **Decision: do not mirror it.** The size was the wrong question. Flickr30k's authors
      state they do not hold copyright in the images, that use is governed by Flickr's own
      Terms of Use, and that the data is provided *solely at their link* for non-commercial
      research and education. No redistribution right is granted, so re-hosting it under
      `roshbeed/` is not mine to do. This is the case the old "don't re-host third-party
      data" note was actually about -- text8 is public domain and CREMA-D is ODbL, which is
      what made mirroring those two right; the licence is the test, not the gigabytes.

      What that leaves unprotected is real and worth naming: `nlphuji/flickr30k` is a
      community account, and the config reads it at `refs/convert/parquet`, a Hub-generated
      branch. If either moves, `make train` for captioning breaks. The shipped *result* does
      not, because `adapter_patches.pth` is under `roshbeed/` and serving never touches
      Flickr30k. Nothing here depends on someone else's account at serving time, which is
      what the rule is for; the training path depends on a dataset nobody may re-host, which
      is a property of the dataset rather than a gap to close.
- [x] `multimodal-captioning` -- SmolLM2-135M as the decoder: **measured, and rejected.**

      Compute first, on the real training step (the service's own `caption_loss` over its
      own adapter and collate function, 8 timed steps after 3 warmup):

      | decoder | batch 4 | batch 8 |
      |---|--:|--:|
      | Qwen3-0.6B-Base (596M, 28 x 1024) | 0.436s* | 0.817s |
      | SmolLM2-135M-Instruct (135M, 30 x 576) | 0.181s | **0.256s** |

      **3.19x at batch 8 against the 2.9x the arithmetic predicted** -- so that estimate
      held, worth recording next to the warning that two earlier ones were out by ~40x.
      (*The Qwen batch-4 mean was 1.173s with a 5.818s max because MPS threw `Discarded
      (victim of GPU error/recovery)` mid-run; its minimum is the usable figure.)

      Then quality, trained at the shipped run's settings -- 5,000 Flickr30k pairs, 15
      epochs, same adapter, same seeded split, same CLIP-agreement harness at 200 pairs:

      | decoder | untrained floor | trained | human ceiling | gap closed |
      |---|--:|--:|--:|--:|
      | Qwen3-0.6B (shipped) | 0.1848 | 0.2115 | 0.3233 | **19.2%** |
      | SmolLM2-135M | 0.1984 | 0.1964 | 0.3286 | **-1.6%** |

      It scores *below its own untrained floor*. 3.19x cheaper per step and it does not
      work at this scale, so **the decoder stays Qwen3-0.6B** and `adapter_patches.pth`
      stays the default. The SmolLM2 adapter is deliberately not registered in
      `config.CHECKPOINTS` and not pushed: that registry drives the frontend's checkpoint
      list, and a 576-wide adapter loaded against a 1024-wide decoder would produce
      confident nonsense.

      **The way it failed is the point, and it is the same failure a second time.**
      Held-out loss fell cleanly for all 15 epochs, 2.7203 to 2.3035, while agreement sat
      below the floor the whole way; the captions say why -- "a black and white tie, and a
      black and white tie". Loss is the wrong scoreboard for this service, which the
      pre-fix Qwen3 run already showed, and this is an independent confirmation with a
      different decoder and a different vocabulary. Note also that the two losses are not
      comparable to each other at all: cross-entropy over SmolLM2's 49,152 tokens is
      naturally lower than over Qwen3's 151,936, so the only number that means anything
      across a decoder swap is agreement.

- [x] `vit-classifier` -- **99.52% on the 10,000-image MNIST test set** against an 11.35%
      majority-class baseline. Two separate lifts. First, 91.78% to 98.86% by running the
      training the config already specified: it said 100 epochs with patience 15 and the
      shipped checkpoint came from a one-epoch smoke run nobody followed up. Then 98.86% to
      99.52% by augmenting position and scale, which was done for robustness and improved
      the clean benchmark as a side effect.

      **MNIST is centred by centre of mass and size-normalised into a 20x20 ink box; nothing
      a person draws is either.** `serving/api.py` plain-resizes whatever the 280x280 canvas
      sends, so position and scale are constants in training and free variables in the only
      input a human produces -- and the model had learned to rely on them rather than ignore
      them. Real MNIST test digits cropped to their ink and redrawn on the canvas, at 60%
      fill: centred 96.5%, a quarter of the way to a corner 66.5%, **halfway 24.5%**, pushed
      up 29.0%. After a `RandomAffine` on the training transform (translate 25%, scale
      0.6-1.4, rotate 12 degrees): 99.8% / 99.8% / **100.0%** / 99.5%.

      `random_split` hands both halves of a split one dataset and so one transform, which
      would have augmented validation too. MNIST is opened twice, once per transform, and a
      seeded permutation indexes into whichever copy each half should see.

      Two things had to be fixed before the run was worth starting. It had no resume, on a
      machine that dies on OOM -- it uses `resume_from` / `save_resume` / `resume_path_for`
      now, with the epoch cap in the fingerprint because the cap is the cosine schedule's
      `T_max`, and the scheduler fast-forwarded so a resumed run does not restart the decay
      from the top. And **the test accuracy it printed was not the shipped checkpoint's**:
      the test pass ran on whatever weights the loop exited with, which under early stopping
      is `patience` epochs past the best one. It reloads the best checkpoint first now.

      Validation loss bottoms at epoch 23 while accuracy keeps climbing to 40, so **loss is
      the wrong thing to early-stop this service on** -- the model is getting more confident
      about what it has right and more wrong about the rest. Accuracy is what ranks the
      checkpoint, so the drift costs nothing. The 114 errors are spread across all ten
      digits (9 worst at 97.92%, 3 best at 99.80%) and the commonest confusions are 9/4 and
      5/3, which is handwriting rather than a systematic failure.
- [x] `vit-seq2seq` -- variable-length sequences, so EOS is a decision rather than a
      position. **93.77% exact-sequence accuracy on 3,000 held-out composites of one to
      three digits**, against 3.33% for always answering the commonest number in the split
      and 0.10% for a uniform guess over 0-999. Per length: 98.60% / 94.50% / 88.20%. **EOS
      lands in the right place 100.00% of the time**, and per-digit accuracy is 96.53%.

      **That is lower than the 98.53% this file used to record, and it describes far more.**
      See the geometry entry below: the old number held only for digits at about a quarter
      of the canvas, and the same checkpoint read 3.3% at 20 pixels and 10.0% at 95. `make evaluate` prints all of
      it, including the per-digit figure, and `make evaluate-sampled` is the same scoring with
      the decode rule flipped; neither existed before, and the old 68.00% came out of a
      training run's validation pass.

      Trained 15 epochs, best at epoch 14 (val 98.33%), so the schedule is spent rather
      than cut short -- the last epoch was slightly worse.

      **Decoding was sampling, and that was worth 21.5 points.** Both decode loops drew
      each token with `torch.multinomial` at temperature 1. On the same checkpoint and
      split, both arms out of `make evaluate`: sampled 77.03% exact / 90.23% EOS / 90.60%
      per digit, greedy **93.77% / 100.00% / 96.53%**. For a task with one right answer,
      sampling picks a low-probability token every so often and loses the sequence with it
      -- one served request read a clear 206 as 22, with the second digit drawn at
      probability 0.007. Greedy is the default now and `sample=True` is opt-in. The length
      spread shows where it hurts: decoding 1,000 one-digit targets, sampling emits nothing
      11 times and two digits 71 times, each one unlucky draw on a distribution the model
      had right.

      It hid because *both* sides agreed. The trainer's validation pass sampled too, so the
      per-epoch number matched what `make evaluate` reported afterwards, and two agreeing
      numbers read as a verified one. Same shape as the retrieval vocabulary bug: training
      and evaluation shared the defect, and only a real request through the container showed
      it -- the same image answered differently on consecutive calls.

      **The training data was not the data the API produces, and that was worth 33.7
      points.** `_load_base_mnist` applied `transforms.Normalize` to each MNIST digit and
      `__getitem__` then normalised the finished composite again, while the API decodes a
      PNG in [0, 1] and normalises once. Training inputs ran -1.80 to 8.73 against the
      API's -0.42 to 2.82, so every stroke arrived at roughly a third of the amplitude the
      model trained on and a composite's background sat at MNIST's mean rather than black.
      Scored on the same 3,000 composites, the then-shipped checkpoint read **97.93%
      exact-sequence on the loader's tensors and 64.23% on the tensors serving builds from
      a faithful PNG of the same images**; correct length 100% against 66.27%. So the
      published figure was measured on the wrong side of a 33.7-point gap, and nothing
      structural could see it -- tests passed, `/health` was green, the loss fell.

      Found the same way as the sampling bug: a real request. After greedy decoding was
      fixed the container still read a clear 206 as `20`, and the local model agreed on the
      same PNG, which ruled out the service and pointed at the image. The fix is
      `transforms.ToTensor()` only when loading MNIST and one normalisation at the end, on
      the finished composite, with the API's own mean and standard deviation -- and then a
      retrain, because it changes the training distribution.

      **The two paths are now measured against each other, not assumed equal.** On the same
      3,000 composites, routing one copy through a real PNG encode and the service's
      `preprocess_image`: identical exact-sequence, identical per-digit, the same answer on
      all 3,000, largest per-pixel difference 0.006 (8-bit PNG quantisation).
      `test_a_composite_survives_the_round_trip_the_api_puts_it_through` fails if that stops
      being true -- it recovers pixels from the training tensor, encodes the PNG a caller
      would post, pushes it through the real `preprocess_image` and compares. Swept the rest
      of the workspace for the same class: captioning routes both paths through
      `backbones.image_processor`, both speech services through `load_model_and_processor`,
      `vit-classifier` normalises once on each side. This was the only hand-rolled
      training/serving preprocessing pair.

      **Three other things capped the old number.** First the EOS design: scored on the
      variable-length task, greedy, the old checkpoint reads 23.20% exact-sequence and
      **0.00% on both one-digit and two-digit targets**, getting the length right 33.33% of
      the time, because `zfill(3)` pinned EOS to position 4 in every training example. It
      emits three digits whatever it sees.

      Second the metric. `sequence_to_number` folded tokens back into an integer *and*
      skipped token 0, so `[4,0,7]` read as 47 and predicting `47` for `407` scored correct.
      A unit test asserted that behaviour. The published 68.00% was never a sequence
      accuracy.

      Third geometry, which is the one that answers "this is MNIST, why is per-digit
      accuracy 79%". The patch embedding is a strided conv, so at patch 16 an 84px canvas
      yields a 5x5 grid covering 80px and silently drops columns 80-83. The third digit sat
      at columns 56-83, so **14% of its width never reached the model**. 112px is 7x16.
      `digit_size` was also `image_size // max_digits`, which forced `spacing` to 0 and so
      `max_jitter = spacing // 2` to 0: the positional jitter the dataset advertised had
      never moved anything.

      96.53% per digit against the sibling classifier's 99.52% on single clean digits. The
      two share an encoder; this one reads glyphs that are sheared, rotated, rescaled and
      placed anywhere on the canvas at any size, which is a harder problem than a clean
      centred 28x28, and the gap is the price of that range.

- [x] `vector-search-retrieval` -- trained with in-batch negatives on 88,523 MS MARCO
      triples. **Recall@10 0.8381 against a 0.0020 chance floor and 0.0014 for untrained
      towers of the same shape**: 419x chance, recall@1 0.5316, **median rank 1 of
      5,000**. Verified through the container against the 104,808-document index compose
      builds, not only in-process. `make evaluate` prints all of it, including the
      vocabulary split, from the shipped checkpoint.

      | loss | recall@10 | recall@1 | median rank | best loss |
      |---|--:|--:|--:|--:|
      | triplet | 0.3047 | 0.0846 | 35 | 0.0025 |
      | **in-batch** | **0.6728** | **0.3778** | **3** | 0.4054 |

      **Both rows of that table are on the text8-only vocabulary**, before the extension
      below. They are comparable to each other, not to the 0.8381 headline; the triplet
      arm has not been re-run on the extended vocabulary.

      **Triplet's final loss is 159x lower than in-batch's and its retrieval is less than
      half as good.** A satisfied margin contributes exactly zero, so an easy negative
      drives triplet's loss toward nothing while teaching the towers nothing. Never rank
      checkpoints on a loss compared across objectives. Same lesson as captioning's, from
      the opposite direction. Pairwise accuracy nearly hid the whole effect too: 0.9770
      against 0.9690, on a metric whose chance is 0.5.

      The published post's 3.06x reproduces as 2.21x here -- direction and order of
      magnitude, not the exact ratio, on a different corpus and pool. Do not quote "3x".

      **The vocabulary was the ceiling, and closing it was worth 0.165 recall@10.** It
      was a property of the representation, not the objective: 1,506 of 5,034 held-out
      queries contained a word the text8-trained vocabulary never saw, scoring 0.4309
      recall@10 against 0.7761 for the 3,528 it covered, and under triplet loss the same
      split was 0.3679 against 0.1567 -- **changing the objective moved both columns and
      closed none of the gap**. `tokenize_text` maps an unknown word to token 0, which is
      also the padding id, so the most discriminative word in a query reached the GRU
      indistinguishable from padding.

      `make extend-vocab` is the fix, and it is the stage `content-scoring` already has:
      count over the **training** triples only (a vocabulary is fitted state), keep every
      pretrained id where it is and append above them, continue-train CBOW from the
      pretrained weights. 63,642 -> 88,114 words, 2 epochs, ~13 minutes each.

      | | queries | recall@10 | median |
      |---|--:|--:|--:|
      | text8 only, all words known | 3,528 | 0.7761 | 2 |
      | text8 only, one or more unknown | 1,506 | 0.4309 | 20 |
      | **extended, all words known** | **4,308** | **0.8935** | **1** |
      | **extended, one or more unknown** | **726** | **0.5096** | **10** |

      **Half the problem was punctuation, not vocabulary**, and that is the part worth
      remembering. text8 is cleaned Wikipedia -- lowercase letters and spaces, no
      punctuation at all -- while MS MARCO is raw prose and the tokenizer split on
      whitespace alone, so `however,` and `cells.` were tokens the vocabulary could not
      contain. **19.4% of document tokens were unknown that way against 8.0% after
      stripping edge punctuation** -- the index itself, not just the query. Stripping
      first also halves what the extension has to add, because the glued variants collapse
      into words already present: 24,472 new words rather than 53,735.

      The covered bucket improved too, 0.7761 -> 0.8935, which is the document side
      getting better for every query rather than only the rare-word ones.

      **A word absent from the new corpus does not move, and that was checked.** CBOW
      reads `in_embed` for context words only, so 15,419 of the 15,420 text8 words MS
      MARCO never uses come out bit-identical. The exception is `<PAD>`, which moves
      because it *is* context -- it fills the window at a document's edges.

      **`/query` and the index tokenized differently, and the served path was at chance.**
      `tokenize_text` built the index and lowercased; `retrieval/query_engine.py:tokenize`
      answered `/query` with `text.strip().split()` and did not. MS MARCO queries ship
      lowercase, so the held-out evaluation never exercised it and every published number
      was measured on a path a person does not use. On the same 5,034 queries and pool:
      lowercase 0.6728, Capitalised First Word 0.5981, **Title Case 0.0020 -- the chance
      floor**, median rank 2,477. A capitalised word missed the vocabulary and arrived as
      the padding id, so a title-cased query was a query of pure padding.

      One `normalise_words` in `cbow/vocabulary.py` now, used by the triples, the index and
      `/query`; all four casings score 0.8381 identically, and
      `test_the_served_tokenizer_matches_the_one_that_built_the_index` fails if they
      diverge. **Second instance of this bug class in one pass** -- vit-seq2seq's double
      normalisation was the first -- which is why the standing rule above exists.

      **The result above is the second attempt, and the first one was wrong.** The first
      run tokenized the triples *before* rebuilding the vocabulary, so
      `tokenized_triples.json` carried ids from the old 7,692-word mapping (max id 7,688)
      while the frozen CBOW matrix the towers index was built for 63,642 words. Every
      embedding lookup returned a different word's vector.

      It was invisible everywhere it was checked. Training and evaluation read the same
      wrong mapping, so the loss fell, recall@10 reproduced at 0.3983 across separate
      runs, `make evaluate` agreed with itself, 31 unit tests passed and `make check` was
      green. **Only serving broke**, because `/query` tokenizes text live and therefore
      used the *right* mapping: in a container the correct document for "where was movie
      the birds filmed" sat in the index scoring cosine 0.049 against its own query while
      unrelated passages ranked above it. Health checks said `checkpoint_loaded: true`
      throughout. Verify by running, and verify the artifacts agree with *each other* --
      "it pulled what I published" is not the same as "what I published is consistent".

      Fixing it was worth 40% of the achievable recall: 0.3983 to 0.6728.

      The guard is exact, because a bound does not work: the stale ids all ran below 7,692
      and so fitted inside the larger vocabulary perfectly while naming different words.
      The file keeps each document's text beside its ids, so verification recomputes them
      -- 200 sampled documents, compared, and it raises rather than warns. Checked against
      the real stale file (rejected, 200/200 disagreed) and the rebuilt one (accepted).
      New files also record a `vocabulary_fingerprint`. Same bug class as the
      word-embeddings `word_to_index` / `word_to_lemma_index` mismatch already noted above;
      it recurred because nothing tied the ids to the mapping that made them.

      **Six files are one artifact set, and partial publication has now broken serving
      twice.** They are `words_to_ids.json`, `ids_to_words.json`, the CBOW matrix, the two
      towers and `tokenized_triples.json` -- the vocabulary maps words to row numbers, the
      matrix has one row per word, each tower carries that matrix frozen inside it, and the
      triples hold ids that index the same mapping. The first failure was the tokenizer and
      tokenize stages writing their outputs locally and uploading nothing, so a rebuild left
      the Hub holding a 63,642-row CBOW checkpoint beside a 7,692-word vocabulary. The
      second was the vocabulary extension: five of the six were published and the triples
      were not, because the tokenize stage had been run with
      `AI_RESIDENCY_HF_DATASET_REPO_ID=` to skip a 317MB upload. **That one looked like
      success from every angle a check was pointed at** -- `/query` answered, every word
      came back known, the embedding norm was healthy -- because `/query` tokenizes live
      and so used the right mapping. What was empty was the *index*: the guard correctly
      refused to build it against stale ids, and compose's `|| echo "index: skipped"`
      swallowed the message. `results: []` for every query, with nothing anywhere saying
      why.

      `make publish` (`data_prep/publish.py`) is the fix: one command that verifies the
      CBOW row count, both towers' embedding widths and the triples' recorded
      `vocabulary_fingerprint` all agree with the vocabulary on disk, and pushes nothing
      unless they do. Compose's failure message now names the artifact-set cause rather
      than blaming an unreachable Hub. **A guard that raises is worth nothing if the
      caller discards its output** -- the `||` branch was swallowing the one line that
      explained a silently broken deployment.

- [x] `rlhf-ppo` -- `make train-loop` exists and the before/after is reported. 60 PPO
      steps on real `CarperAI/openai_summarize_tldr` prompts, 16 held-out prompts never
      trained on, baseline before any gradient step, fresh adapter:

      | | greedy | sampled (4x, seeded) | policy drift |
      |---|--:|--:|--:|
      | pre-PPO | -3.4202 | -5.0505 | 0 |
      | after 60 steps | -3.4585 | -4.9640 | 1.26e-04 |

      **The two metrics disagree about the sign and that is the result** -- greedy -0.04,
      sampled +0.09, on a policy whose weights moved a hundredth of a percent. Neither
      direction is a finding at that magnitude, and quoting one alone would invent one.

      **A bigger learning rate is ruled out, not untried.** Scaling the trained adapter's
      B matrices scales the drift without retraining, and reward falls monotonically:
      greedy -3.4585 at the actual 1.26e-04, -3.7579 at 1.26e-03, -4.9454 at 1.26e-02,
      -5.6364 at 1.26e-01. So the metric is not blind (it responds sharply from 1e-03 up,
      meaning the flat reward really is a policy that did not move), and travelling
      further in *this* direction is worse. Larger steps would find a different direction,
      so this kills the cheap explanation rather than the whole idea; the suspects left
      are advantages from 4-prompt batches, 10-token completions, and the missing SFT
      reference.

      **Three bugs, none visible to a passing test.** `_load_ckpt` had never loaded a
      checkpoint: `load_checkpoint` returns the saved *record* and it was passed whole to
      `load_state_dict(strict=False)`, so none of the five wrapper keys matched, nothing
      loaded, and it logged `Loaded policy LoRA checkpoint (513 missing, 5 unexpected)`.
      The 5 was the whole story. Fixing it then made the loop resume its own previous
      adapter and call it "pre-PPO", so `run_loop` passes `load_adapters=False` and checks
      baseline drift is 0. And §6 claimed the reward model was an untrained Qwen3 head when
      it is `OpenAssistant/reward-model-deberta-v3-large-v2`, inference-only.

      Log a magnitude beside a metric that did not move. "Reward did not change" and "the
      metric cannot see the change" are indistinguishable without one, and greedy decoding
      reported the identical reward to four decimal places at iterations 10, 20, 30 and 40
      because an argmax has to flip before it says anything.
- [x] `whisper-fine-tuning` -- meaningful by its own design and needs no training run. It
      fine-tunes on a single clip so the effect of a gradient step is visible rather than
      statistical: `"My name is Roosh."` becomes `"My name is Rosh"`, with the base model's
      own output as the baseline.

      The counterweight is measured now, and it did not come out the comfortable way.
      `make collateral` transcribes twelve CREMA-D sentences the model never trained on,
      before and after: **one step leaves WER at 0.1642 and alters no transcript by a word;
      five steps takes it to 0.2687 and alters four of twelve** -- "The airplane is almost
      full" to "The airplay is almost full", "Don't forget a jacket" to "Go for it". So the
      overfitting this service demonstrates is not only visible on the clip it trained on,
      and the write-up says so.

      CREMA-D is the corpus for it because its twelve sentences are encoded in every
      filename, so the reference comes free. WER is over normalised words: the target text
      has no final period, the model stops emitting one after the step, and every raw
      transcript then differs while no word does.

      That also found a disagreement worth not repeating. `make demo` takes `num_epochs=1`
      while `/transcribe` hardcoded `num_steps=5` twice, so the demo and the endpoint were
      doing different things and the README's headline claimed the five. It reads
      `serve_num_steps` now, with the measured cost in the field's docstring.

      **`serve_num_steps` is 3, because the middle of the curve got measured.** Five was
      chosen from two data points. `make collateral` at every count in between: WER 0.1642
      / 0.1493 / 0.1493 / 0.1642 / 0.2687 at one through five steps, altering 0 / 2 / 2 / 2
      / 4 of twelve. The altered count is the better signal -- through four steps the two
      that change were *already wrong* at base, and only at five do transcripts the base
      model got right start breaking. Three keeps a streamed trajectory, corrects the demo
      clip, scores below the base rate and breaks nothing that worked. **Measuring the ends
      of a range and interpolating is not measuring it.**

### Container verification, 2026-10-18: all nine, with a real request each

Not `/ready` -- a real call, because this pass learned twice over that a service can
report `checkpoint_loaded: true` and still be wrong. word-embeddings (`computer` ->
computers, hardware, computing, software), content-scoring (a Show HN title scored 2.09
with a word trace), vit-classifier (a drawn `1` at 0.958), vit-seq2seq (three drawn digits
decoded), speech-emotion-recognition (**4/4** held-out CREMA-D clips correct above 0.99),
whisper-fine-tuning (five streamed steps, loss 7.1992 -> 0.3033, corrected at step one),
rlhf-ppo (`/generate` with per-token traces), multimodal-captioning (Flickr30k images,
people-and-clothing captions matching §6), vector-search-retrieval (correct document 3rd
of 105,052), and the frontend.

Re-verified for the three services that changed since: `vit-seq2seq` (1-, 2- and 3-digit
composites read as 0, 55 and 206 at ~0.92 per digit, five identical requests giving five
identical answers), `vit-classifier` (three held-out test images at confidence 1.00), and
`vector-search-retrieval` (correct document **1st** of 104,808, `urinalysis` now finding
urinalysis passages where it used to return a thermocouple, and a title-cased query giving
byte-identical results). 363.8 MiB resident for retrieval after a real query.

**`--pull always` is for verification too, not just the quickstart.** Checking captioning
without it, `docker run` served a local `latest` from days earlier whose config still had
`DEFAULT_CHECKPOINT = "adapter_flickr_batched.pth"`. The captions came back degenerate --
"The 2010 World Cup Final between Brazil and Argentina" -- and for a few minutes that
looked like the shipped service defaulting to the wrong adapter. Pulling and re-running
gave `adapter_patches.pth` and sensible captions. The note above about `docker run` reusing
a stale tag is in this file already; it still caught me, in the one place where being
caught produces a false alarm about the thing being verified.

## Next pass: optimization, once every service is at a meaningful checkpoint

The pass above gets each service to a result that beats a stated baseline. This one makes
those results better, and it is deliberately narrow about how.

**What is in scope:** size and shape. Hyperparameters, how long something trains, how much
data it sees, the width or depth of the one trainable part, swapping a pretrained model for
a different pretrained model, changing a loss function, changing what gets measured and
how. Quick and useful, not sprawling -- the point is a better number for an afternoon's
work, not a research programme.

**What is not:** the concept or the architecture. Late fusion stays late fusion. Attention
stays written out rather than called. The adapter stays the only thing that trains in
captioning. Emotion stays a token the decoder emits. The two towers stay two towers. If a
change would make the service demonstrate a different thing, it belongs in a different
service, not in this pass.

**Measure the lever before spending an afternoon on it**, which is the one rule the last
pass earned the hard way. Three of this pass's leads below are already measured, and two
things that looked obvious turned out to be wrong when checked: a smaller captioning
decoder is 3.19x cheaper and scores *below its own untrained floor*, and a larger PPO
learning rate makes reward monotonically worse rather than better. Both would have been a
wasted afternoon and a wrong write-up.

Leads, strongest first, with what is already known:

- **`vector-search-retrieval`: done, and it was the biggest lever in the repo.** Extending
  the text8 vocabulary onto MS MARCO took recall@10 **0.6728 -> 0.8381**, recall@1 0.3778 ->
  0.5316 and the median rank from 3 to 1. The towers did not change. Details in the status
  entry above, including the two things worth carrying elsewhere: half the gap was
  punctuation rather than vocabulary, and `/query` was answering at the chance floor for any
  capitalised query because the served tokenizer and the one that built the index disagreed.
- **`content-scoring`: the encoder, measured.** A mean of static word vectors throws away
  word order; a fine-tuned MiniLM already beats the shipped model by +0.017 rho and +19% on
  finding front-page posts, and capacity over the *fixed* representation does nothing. The
  probes are in `notebooks/content-scoring/`. This is a model swap, which is in scope, and
  the number to beat already exists.
- **`rlhf-ppo`: the signal, not the step size.** Scaling the update 1000x makes reward
  monotonically worse, so the learning rate is ruled out. What is left: advantages
  estimated from 4-prompt batches, 10-token completions, and the missing SFT stage (the KL
  reference is the base model, so the penalty measures drift from the wrong thing). Also
  unmeasured: `OpenAssistant/reward-model-deberta-v3-base` is 3.5x cheaper in compute than
  the large-v2 currently used and inference-only, so it may not be the bottleneck at all.
- **`vit-seq2seq`: reopened, then closed properly.** 93.77% exact-sequence, 96.53% per
  digit, EOS correct on every held-out sequence, across every scale a person can draw at.
  The previous 98.53% was real and useless: it held only for digits at one size, and the
  service read **10.0% on a single digit drawn large, which is the only thing the frontend
  canvas invites you to do**. The composite geometry is augmented now, not just the glyphs
  inside it. The last epoch of 15 was still the best, so there is a little left in longer
  training.

  **Check the decode rule before the model, in any service that generates.** Sampling at
  temperature 1 cost this service 21.5 points of exact-sequence accuracy, and the
  train/serve normalisation skew cost another 33.7 -- between them more than every other
  fix here put together, and the first was one `argmax` away. `multimodal-captioning` already
  decodes greedily and measured `no_repeat_ngram_size=3` as a wash; `rlhf-ppo` reports both
  greedy and seeded-sampled reward on purpose. This was the one place a sampled decode was
  being read as a deterministic answer.
- **`vit-classifier`: done, twice.** One epoch at 91.78% to **99.52%**, with no change to
  the architecture. The first lever was not a hyperparameter at all -- the config already
  specified 100 epochs with patience 15, and the shipped checkpoint came from a one-epoch
  smoke run that was never followed up, so **check what the config already says before
  tuning it.** The second was position and scale augmentation, taken on for robustness
  against hand-drawn input and worth +0.66 on the clean benchmark as well: a digit drawn
  halfway to a corner went 24.5% to 100.0%.
- **`whisper-fine-tuning`: done.** `serve_num_steps` is 3. The full curve at one through
  five steps says five was the only count doing real damage, and three scores *below* the
  base model's WER while keeping a streamed trajectory. See the status entry above.
- **`speech-emotion-recognition`: already known.** Held-out loss bottoms at epoch 6 and
  accuracy peaks at 10 against the 20 it ran, so the useful range is eight to ten. Cheaper
  rather than better.
- **`multimodal-captioning`: closed for now.** The decoder swap is measured and rejected,
  and §6 already shows 16x the data buys almost nothing. The adapter's own width and depth
  are the only untested knob, and it is the one thing that trains.

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
  **0.7064 held-out accuracy against a 0.1708 majority-class baseline**, 4.1x chance over six
  emotions, and 20/30 held-out clips correct through the container. It is also robust to level
  and room noise now, which the status entry above covers; the epoch question is settled by
  early stopping, which lands at 8.

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
- `content-scoring` ships the `poisson` objective, not `log_mse` -- the note here said otherwise
  until the served checkpoint was read: `extra.objective` is `poisson` and its `total_ratio` is
  **0.9267**, so it predicts 93% of the upvotes that actually happen rather than the fifth `log_mse`
  managed. The API's `model_info` string said "Poisson objective" correctly all along. A binary head
  on `score >= 100` still finds front-page posts best and cannot rank at all, so that trade is open;
  the count-prediction half of it is settled.
- The eleven predecessor repos are archived on GitHub. The four `backup/*` tags that held the
  pre-rewrite tips are deleted from the remote now the repo is public; they survive locally only.
