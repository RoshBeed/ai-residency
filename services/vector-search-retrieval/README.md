# Dual-Encoder Document Retrieval

Container source for dual-encoder document retrieval: a two-tower dual-encoder that retrieves
documents from MS MARCO by approximate nearest-neighbour search. CBOW word embeddings feed a GRU
query tower and a GRU document tower trained with a triplet loss; the trained document tower embeds
the corpus once, offline, into a
**Redis** index (the RediSearch module's HNSW ANN search), and a query only needs one forward pass
through the query tower at request time. Redis is a real runtime dependency, not optional — the
`serve` and `train` containers both `depends_on` it. The full write-up — architecture diagrams,
the triplet-loss derivation, and honest experimental results (52.0% pairwise accuracy against a
50.2% untrained baseline) — lives in
[`ai-residency-frontend/docs/dual-encoder-retrieval.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/dual-encoder-retrieval.md),
not here.

## Prerequisites

- Python >= 3.11, [`uv`](https://docs.astral.sh/uv/)
- A running RediSearch-capable Redis (`make redis` starts a throwaway one on `:6390`)
- Docker, for the container targets

## Commands

```bash
make install        # uv sync
make redis          # throwaway redis-stack-server on :6390

make train           # train-cbow, train-triples, train-encoder, in order
make train-cbow      # build the vocabulary, train CBOW word embeddings
make train-triples    # build and tokenize MS MARCO triples
make train-encoder    # train the GRU dual-encoder towers

make index           # embed the corpus with the trained doc tower, index it in Redis
make query            # index the corpus, then the interactive query CLI (needs a tty)
make serve            # FastAPI backend on :8000 (needs Redis running and a populated index)

make test             # pytest
make check            # lint + strict mypy + tests
```

`make train-cbow` picks up Apple Silicon's MPS backend automatically where it helps — benchmarked
~3.7x faster than CPU on this repo's own batch/vocab shape. The GRU tower training in
`train-encoder` trains one triple at a time on tiny tensors and benchmarked *slower* on MPS due to
per-op dispatch overhead, so it stays CPU-only intentionally. `config/sweep.yaml` runs a W&B sweep
over the same `Config` fields `make train-encoder` uses.

`make query` is the interactive retrieval demo: it indexes the corpus, then drops into a query
loop on stdin. It needs an attached terminal, which is why it's a local Makefile target rather than
a Docker service — the real persistent HTTP server is `make serve`.

## Docker

Two single-purpose images, both `FROM roshbeed/ai-residency-base:local` (built by
`ai-residency-repo-template`'s `Dockerfile.base` — build that once first with `make docker-build-base` over
there, or pull it once it's published). Neither of these two ever touches git or needs a
credential: `ai-residency-common` is already installed in the base image by the time their own `uv sync`
runs.

- **`Dockerfile.serve`** — the FastAPI retrieval service Kubernetes runs.
- **`Dockerfile.train`** — the offline pipeline (CBOW embeddings, triples, tower training) a
  RunPod/AWS GPU host pulls and runs. Runs the full pipeline by default; override `CMD` to run a
  single stage, e.g. `docker compose --profile train run --rm train uv run python -m cbow.train_tokenizer`.

```bash
make docker-build-train    # docker build -f Dockerfile.train
make docker-build-serve    # docker build -f Dockerfile.serve

make docker-up             # redis + serve, via docker compose
make docker-train           # run the full training pipeline in the train container
make docker-down            # stop and remove all compose services

make docker-push-train      # push roshbeed/ai-residency-week2-train (needs `docker login`)
make docker-push-serve      # push roshbeed/ai-residency-week2-serve
```

`docker-compose.yml` has two services: `serve` (default) and `train` (profile-gated, so a plain
`docker compose up` never kicks off a training run). `redis` is a `redis-stack-server`, no
persistence, throwaway by design. Both `serve` and `train` `depends_on: [redis]`.

## Environment variables

Everything degrades rather than failing, so a fresh clone still runs without any of these:

| Variable | Absent |
|---|---|
| `HF_TOKEN` | Checkpoints stay local, warning logged; serving falls back to a local checkpoint only |
| `WANDB_API_KEY` | Run goes offline, sync later with `wandb sync` |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | Defaults to `localhost:6390`, no password — matches `make redis` |

`REDIS_HOST=redis` inside Docker Compose, pointing at the `redis` service by name instead of
`localhost`. Any field on `Config` (`config/`) can
be overridden with an `AI_RESIDENCY_<FIELD_NAME>` environment variable — e.g. `AI_RESIDENCY_NUM_EPOCHS=1 make
train-encoder` for a fast smoke run. A `.env` in this repo or in the parent directory (checked
first, nearest wins) is loaded automatically; see `.env.example`.

## Standards

Shared behaviour — device selection, logging, checkpoint format, seeding, Hub upload, W&B
tracking — comes from [`ai-residency-common`](https://github.com/RoshBeed/ai-residency-common), pinned by tag. Ruff
(format and lint) and mypy run in strict mode with zero suppressions; the `[tool.ruff]`/
`[tool.mypy]` blocks are copied verbatim from `ai-residency-common` so configuration cannot drift between
repos in this portfolio.

## Full write-up

Architecture diagrams, the method (triplet loss, MS MARCO triple construction, CBOW word
embeddings), and experimental results live in
[`ai-residency-frontend/docs/dual-encoder-retrieval.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/dual-encoder-retrieval.md).
