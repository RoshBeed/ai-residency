# Content Scoring (regression head over frozen word embeddings)

Container source for a content-score predictor: a small MLP that estimates an engagement
(upvote) score from a post's title, built on top of word embeddings trained from scratch in the
sibling [`ai-residency-week1-word-embeddings`](https://github.com/RoshBeed/ai-residency-week1-word-embeddings)
repo. The word vectors are frozen and pulled from that repo's Hugging Face repo; this repo only
trains the regression head. The full write-up — architecture, the honest result that the trained
head does not clearly beat a predict-the-mean baseline on 782 labelled posts, and the reasoning —
lives in
[`ai-residency-frontend/docs/content-scoring.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/content-scoring.md),
not here. This repo is container source, not documentation.

## Layout

A `data → model → train → evaluate` pipeline over frozen vectors in the `content_scoring/`
package, plus a thin serving layer at the root:

| File | What it does |
|---|---|
| `content_scoring/embeddings.py` | Load the frozen CBOW matrix and vocabulary from the sibling repo's checkpoint |
| `content_scoring/data.py` | Turn each post title into an averaged word vector (`TitleScoreDataset`) |
| `content_scoring/model.py` | `ScorePredictor` — the small MLP that predicts scores |
| `content_scoring/train.py` | Train the scorer and save the best checkpoint |
| `content_scoring/evaluate.py` | Test MSE against the predict-the-mean baseline |
| `config/__init__.py` | Every hyperparameter for the scoring stage |
| `content_scoring/predict.py` | Score a title from the command line (`python -m content_scoring.predict`) |
| `serving/api.py` | Expose prediction as a FastAPI API (`python -m serving.api`) |

## Prerequisites

- Python >= 3.11, [`uv`](https://docs.astral.sh/uv/)
- Docker, for the container targets
- A trained CBOW checkpoint — pulled from the word-embeddings repo's Hugging Face repo, or present
  locally at `checkpoints/cbow/cbow_model.pt`

## Commands

```bash
make install        # uv sync
make train          # train the regression head (over the frozen CBOW vectors)
make serve          # FastAPI backend on :8000 (POST /predict, GET /health)
make predict TITLE="your title here"   # one-shot score from the command line

make test           # pytest
make check          # lint + strict mypy + tests
```

`make train` writes `checkpoints/best_predictor.pth`, the checkpoint `serve` loads by default. The
frozen CBOW vectors it consumes are never modified.

## Docker

Two single-purpose images, both `FROM roshbeed/ai-residency-base:local` (built by
`ai-residency-repo-template`'s `Dockerfile.base` — build that once first with `make docker-build-base` over
there, or pull it once it's published). Neither ever touches git or needs a credential:
`ai-residency-common` is already installed in the base image by the time their own `uv sync` runs.

- **`Dockerfile.serve`** — the FastAPI scoring service Kubernetes runs.
- **`Dockerfile.train`** — trains the regression head; what a RunPod/AWS GPU host pulls and runs.

```bash
make docker-build-train    # docker build -f Dockerfile.train
make docker-build-serve    # docker build -f Dockerfile.serve

make docker-up             # serve, via docker compose
make docker-train          # train, in a container (profile-gated)
make docker-down           # stop and remove all compose services

make docker-push-train     # push roshbeed/ai-residency-week1-content-scoring-train
make docker-push-serve     # push roshbeed/ai-residency-week1-content-scoring-serve
```

## Environment variables

Everything degrades rather than failing, so a fresh clone still runs without any of these:

| Variable | Absent |
|---|---|
| `HF_TOKEN` | Checkpoints stay local, warning logged; serving falls back to a local checkpoint only |
| `WANDB_API_KEY` | Run goes offline, sync later with `wandb sync` |

Any field on `Config` (`config/`) can be overridden with an `AI_RESIDENCY_<FIELD_NAME>` environment
variable — e.g. `AI_RESIDENCY_NUM_EPOCHS=1 make train` for a fast smoke run. A `.env` in this repo or
in the parent directory (checked first, nearest wins) is loaded automatically; see `.env.example`.

## Standards

Shared behaviour — device selection, logging, checkpoint format, seeding, Hub upload, W&B
tracking — comes from [`ai-residency-common`](https://github.com/RoshBeed/ai-residency-common), pinned by tag. Ruff
(format and lint) and mypy run in strict mode with zero suppressions; the `[tool.ruff]`/
`[tool.mypy]` blocks are copied verbatim from `ai-residency-common` so configuration cannot drift between
repos in this portfolio.

## Full write-up

Architecture, the method (averaging word vectors, the regression head), and the honest experimental
results live in
[`ai-residency-frontend/docs/content-scoring.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/content-scoring.md).
