# Word Embeddings (CBOW, trained from scratch)

Container source for word2vec CBOW embeddings trained from scratch: the ETL stage builds a merged
corpus and vocabulary, then CBOW with negative sampling learns 128-dimensional word vectors. The
downstream consumer of these frozen vectors — the content-score regression head — lives in the
sibling [`ai-residency-week1-content-scoring`](https://github.com/RoshBeed/ai-residency-week1-content-scoring)
repo, which pulls this repo's published checkpoint. The full write-up (method, architecture
diagrams, results, references) lives in
[`ai-residency-frontend/docs/word-embeddings.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/word-embeddings.md).
The Streamlit UI also moved there, as a thin HTTP client of the nearest-neighbour API this repo
serves.

## Layout

One self-contained `data → model → train → evaluate` pipeline in the `word_embeddings/` package,
plus a thin serving layer at the root:

| File | What it does |
|---|---|
| `word_embeddings/data.py` | Build the corpus (HN + text8 ETL, vocabulary) and the negative-sampling dataset |
| `word_embeddings/model.py` | `CBOWNS` — CBOW + negative sampling (read this for the word2vec math) |
| `word_embeddings/train.py` | Train the embeddings and save the best checkpoint |
| `word_embeddings/evaluate.py` | Analogy + similarity checks, t-SNE plot |
| `word_embeddings/etl.py` | Corpus construction: download text8, merge HN titles, lemmatise |
| `config/__init__.py` | Every hyperparameter for the embedding stage |
| `serving/api.py` | Expose the vectors as a FastAPI nearest-neighbour/analogy API (`python -m serving.api`) |

Read `word_embeddings/model.py` for the word2vec math; everything else is plumbing around it.

## Prerequisites

- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- Docker, for the container path
- `roshbeed/ai-residency-base:local` built locally (source: `ai-residency-repo-template/Dockerfile.base`) before
  building either Dockerfile here — both build `FROM` it

## Local development

```bash
make install              # uv sync
cp .env.example .env      # HF_TOKEN, WANDB_API_KEY — both optional, see below

make train-corpus         # stage 1: build the merged vocabulary/corpus
make train-cbow           # stage 2: pretrain CBOW embeddings
make train                # both, back to back (multi-hour)

make serve                 # FastAPI on :8000 (POST /nearest, POST /analogy)
make test                  # pytest
make check                 # lint + strict mypy + tests
```

`make help` lists every target, including `train-cbow-sample` (a tiny built-in corpus, seconds)
and `sweep` (a W&B hyperparameter sweep agent).

## Docker

Two single-purpose Dockerfiles, both `FROM roshbeed/ai-residency-base:local`, replacing what used to be one
multi-stage file:

- **`Dockerfile.serve`** — the FastAPI nearest-neighbour/analogy container. What Kubernetes runs.
- **`Dockerfile.train`** — trains CBOW by default; override `CMD` for the corpus stage
  (`python -m word_embeddings.etl`).

Neither needs a GitHub credential: `ai-residency-common` is already installed in `roshbeed/ai-residency-base`, so
`uv sync` here matches the already-satisfied lock entry instead of re-fetching it.

```bash
make docker-build-train    # docker build -f Dockerfile.train -t ai-residency-week1-train .
make docker-build-serve    # docker build -f Dockerfile.serve -t ai-residency-week1-serve .

make docker-up             # docker compose up --build serve
make docker-train          # docker compose --profile train run --rm train   (defaults to cbow)
docker compose --profile train run --rm train uv run python -m word_embeddings.etl

make docker-down
```

`make docker-push-train` / `make docker-push-serve` push to `roshbeed/ai-residency-week1-train` /
`roshbeed/ai-residency-week1-serve` (need `docker login` first).

## Environment variables

Everything degrades rather than failing, so a fresh clone runs without any of these:

| Variable | Absent |
|---|---|
| `HF_TOKEN` | Checkpoints stay local; serving falls back to whatever is bind-mounted under `checkpoints/` |
| `WANDB_API_KEY` | Training runs offline, sync later with `wandb sync` |
| `HN_DATABASE_URL` | The corpus stage falls back to the last cleaned extract in `data/` if the default mirror is unreachable |

`AI_RESIDENCY_<FIELD_NAME>` overrides any field on `Config` (`config/`) without editing code, e.g.
`AI_RESIDENCY_CBOW_NUM_EPOCHS=1 AI_RESIDENCY_CBOW_EMBED_SIZE=64 make train-cbow`.

## Standards

Shared behavior — device selection, logging, checkpoint format, seeding, Hub upload and W&B
tracking — comes from [`ai-residency-common`](https://github.com/RoshBeed/ai-residency-common), pinned by tag. Ruff
and mypy run in strict mode with zero suppressions; config is copied verbatim from `ai-residency-common` so
it can't drift. Full detail in
[`ai-residency-frontend/docs/word-embeddings.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/word-embeddings.md).
