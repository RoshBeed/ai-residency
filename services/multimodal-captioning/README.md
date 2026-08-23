# Image Captioning (CLIP + Qwen3)

Container source for image captioning (CLIP + Qwen3): bridging a frozen CLIP ViT-B/32 vision
encoder to a frozen Qwen3-0.6B-Base language model. A small trainable adapter (~1.58M parameters)
projects a CLIP image embedding into Qwen3's embedding space as a single "soft token," concatenated
with a text prompt before Qwen3 decodes a caption autoregressively — the same shape LLaVA and
PaLiGemma use. The repo also has a standalone warm-up part (`zero_shot.py`) doing CLIP zero-shot
classification, no training involved. The full write-up — architecture diagrams, the
training-signal derivation, and honest experimental results (real captions from all three shipped
checkpoints, none of them good) — lives in
[`ai-residency-frontend/docs/image-captioning.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/image-captioning.md),
not here. This repo is container source, not documentation.

## Prerequisites

- Python >= 3.11, [`uv`](https://docs.astral.sh/uv/)
- Docker, for the container targets

## Commands

```bash
make install       # uv sync
make zero-shot      # Part 1 — CLIP zero-shot classification over data/images/, ~10s, no GPU needed
make train           # Part 2 — fit the adapter on Flickr30k
make serve            # FastAPI backend on :8000 (POST /caption, GET /health)

make test           # pytest
make check           # lint + strict mypy + tests
```

`make train` writes to `checkpoints/adapter_flickr_batched.pth`, the checkpoint `serve` loads by
default. Running it overwrites that file — point `checkpoint_dir` or `checkpoint_name` elsewhere
first if you want to keep the current one.

## Docker

Two single-purpose images, both `FROM roshbeed/ai-residency-base:local` (built by
`ai-residency-repo-template`'s `Dockerfile.base` — build that once first with `make docker-build-base` over
there, or pull it once it's published). Neither ever touches git or needs a credential:
`ai-residency-common` is already installed in the base image by the time their own `uv sync` runs. Both add
`build-essential`/`libjpeg-dev`/`zlib1g-dev` on top of the shared base — this repo's own
requirement, for Pillow and torch native extensions.

- **`Dockerfile.serve`** — the FastAPI captioning service Kubernetes runs.
- **`Dockerfile.train`** — fits the adapter on Flickr30k; what a RunPod/AWS GPU host pulls and
  runs.

```bash
make docker-build-train    # docker build -f Dockerfile.train
make docker-build-serve    # docker build -f Dockerfile.serve

make docker-up             # serve, via docker compose
make docker-train           # train, in a container (profile-gated)
make docker-down            # stop and remove all compose services

make docker-push-train      # push roshbeed/ai-residency-week4-train (needs `docker login`)
make docker-push-serve      # push roshbeed/ai-residency-week4-serve
```

`docker-compose.yml` has two services: `serve` (default) and `train` (profile-gated, so a plain
`docker compose up` never kicks off a training run). `checkpoints/` and `data/` are bind-mounted at
run time rather than baked into either image.

## Environment variables

Everything degrades rather than failing, so a fresh clone still runs without any of these:

| Variable | Absent |
|---|---|
| `HF_TOKEN` | Checkpoints stay local, warning logged; serving falls back to a local checkpoint only |
| `WANDB_API_KEY` | Run goes offline, sync later with `wandb sync` |

Any field on `Config` (`config/`) can be
overridden with an `AI_RESIDENCY_<FIELD_NAME>` environment variable — e.g. `AI_RESIDENCY_NUM_EPOCHS=1 make train` for
a fast smoke run, or `AI_RESIDENCY_NUM_SAMPLES=500` for a bigger one. A `.env` in this repo or in the parent
directory (checked first, nearest wins) is loaded automatically; see `.env.example`.

## Known limitation

One of the three checkpoints (`adapter_flickr_batched.pth`) was trained with a trailing
`LayerNorm` layer that isn't present for the other two. Checkpoints now record which architecture
produced them (`Config.CHECKPOINTS`, a `final_layer_norm` flag on each entry), so the loader reads
the shape off the artifact instead of guessing — but the underlying captions are still poor across
all three checkpoints, mostly because none of them trained on enough data (10-100 pairs) to move a
596M-parameter frozen decoder's priors. See the full write-up linked above for the actual numbers
and the honest failure analysis.

## Standards

Shared behaviour — device selection, logging, checkpoint format, seeding, Hub upload, W&B
tracking — comes from [`ai-residency-common`](https://github.com/RoshBeed/ai-residency-common), pinned by tag. Ruff
(format and lint) and mypy run in strict mode with zero suppressions; the `[tool.ruff]`/
`[tool.mypy]` blocks are copied verbatim from `ai-residency-common` so configuration cannot drift between
repos in this portfolio.

## Full write-up

Architecture diagrams, the method (the CLIP/Qwen3 bridge, the adapter design, the training signal),
and experimental results live in
[`ai-residency-frontend/docs/image-captioning.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/image-captioning.md).
