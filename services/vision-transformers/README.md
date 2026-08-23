# Vision Transformer Digit Recognition

Container source for Vision Transformer digit recognition: a from-scratch Vision Transformer (patch
embedding, multi-head attention, encoder and decoder blocks, no `torch.nn.Transformer` shortcuts),
applied to two related digit-recognition tasks. An encoder-only classifier reads a single MNIST
digit; an encoder-decoder model reads a composited image of three digits side by side and generates
the number one token at a time, autoregressively. Both models are served through one FastAPI app
that loads both checkpoints at startup. The full write-up — architecture diagrams, the attention
math, and honest experimental results (91.78% classifier test accuracy, 68.00% seq2seq
exact-sequence accuracy, three real bugs found and fixed along the way) — lives in
[`ai-residency-frontend/docs/vision-transformer.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/vision-transformer.md),
not here.

## Prerequisites

- Python >= 3.11, [`uv`](https://docs.astral.sh/uv/)
- Docker, for the container targets

## Commands

```bash
make install           # uv sync

make train-classifier  # train the single-digit encoder-only classifier
make train-seq2seq     # train the multi-digit encoder-decoder model
make train              # alias for train-classifier, the faster of the two

make serve              # FastAPI backend on :8000 (loads both committed/Hub checkpoints)

make test               # pytest
make check              # lint + strict mypy + tests
```

Checkpoints are gitignored and hosted on the Hugging Face Hub, so `make serve` pulls them rather
than needing anything trained first. Where a local checkpoint exists (after `make train`),
`serving/api.py` prefers it.

## Docker

Two single-purpose images, both `FROM roshbeed/ai-residency-base:local` (built by
`ai-residency-repo-template`'s `Dockerfile.base` — build that once first with `make docker-build-base`
over there, or pull it once it's published). Neither of these two ever touches git or needs a
credential: `ai-residency-common` is already installed in the base image by the time their own `uv sync`
runs.

- **`Dockerfile.serve`** — the FastAPI backend Kubernetes runs, serving both models.
- **`Dockerfile.train`** — what a RunPod/AWS GPU host pulls and runs. Trains the ViT classifier
  by default; override `CMD` to train the seq2seq model instead, e.g.
  `docker compose --profile train run --rm train uv run python -m models.seq2seq.trainer`.

```bash
make docker-build-train    # docker build -f Dockerfile.train
make docker-build-serve    # docker build -f Dockerfile.serve

make docker-up              # serve, via docker compose
make docker-train           # train the ViT classifier in the train container
make docker-down            # stop and remove all compose services

make docker-push-train      # push roshbeed/ai-residency-week3-train (needs `docker login`)
make docker-push-serve      # push roshbeed/ai-residency-week3-serve
```

`docker-compose.yml` has two services: `serve` (default) and `train` (profile-gated, so a plain
`docker compose up` never kicks off a training run).

## Environment variables

Everything degrades rather than failing, so a fresh clone still runs without any of these:

| Variable | Absent |
|---|---|
| `HF_TOKEN` | Checkpoints stay local, warning logged; serving falls back to a local checkpoint only |
| `WANDB_API_KEY` | Run goes offline, sync later with `wandb sync` |

Any field on `Config` (`config/`) can be overridden with an
`AI_RESIDENCY_<FIELD_NAME>` environment variable — e.g. `AI_RESIDENCY_NUM_EPOCHS=1 make train-classifier` for a fast
smoke run. The one exception is the classifier's own epoch cap: it's a separate field,
`classifier_epochs`, so it needs `AI_RESIDENCY_CLASSIFIER_EPOCHS` specifically — `AI_RESIDENCY_NUM_EPOCHS` alone
won't shorten it. A `.env` in this repo or in the parent directory (checked first, nearest wins)
is loaded automatically; see `.env.example`.

## Standards

Shared behaviour — device selection, logging, checkpoint format, seeding, Hub upload, W&B
tracking — comes from [`ai-residency-common`](https://github.com/RoshBeed/ai-residency-common), pinned by tag. Ruff
(format and lint) and mypy run in strict mode with zero suppressions; the `[tool.ruff]`/
`[tool.mypy]` blocks are copied verbatim from `ai-residency-common` so configuration cannot drift between
repos in this portfolio.

## Full write-up

Architecture diagrams, the method (patch embedding, multi-head attention, the encoder-decoder
seq2seq design), and experimental results live in
[`ai-residency-frontend/docs/vision-transformer.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/vision-transformer.md).
