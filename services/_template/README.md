# AI Residency Repo Template

The shape every repo in my [Machine Learning Institute](https://ml.institute/) residency
portfolio conforms to. It is a working repo, not a sketch: `make train` trains a small model on
synthetic data, `make serve` answers requests, and `make check` passes. That matters, because a
template nobody has run is a template that doesn't work.

This repo is also the **source of `roshbeed/ai-residency-base`**, the Docker image every week's own
`Dockerfile.train`/`Dockerfile.serve` builds `FROM`. Explanatory write-ups, diagrams and the UI all
live in [`ai-residency-frontend`](https://github.com/RoshBeed/ai-residency/tree/main/services/frontend) now, not
here — this repo (and every week repo shaped like it) is container source, not documentation.

## Instantiating it

1. Copy the tree.
2. Rename the `template/` package to your repo's topic (e.g. `word_embeddings/`, `captioning/`) and
   update the two `from template.… import` sites plus the `python -m template.training` entrypoint.
3. Rename `[project].name` in `pyproject.toml`.
4. Set `hf_repo_id` and `wandb_project` in `Config`.
5. Replace `template/data.py`, `template/models.py` and the body of `template/training.py` with the
   real work. Keep the *shape* of `train()`.

## Layout

```
config/__init__.py       Config(BaseConfig) — every hyperparameter, one dataclass
template/               the package — rename it to the repo's topic on instantiation
├── data.py             build_datasets(config) -> (train, val)
├── models.py           model classes
└── training.py         train(config) -> best_loss, plus main()
serving/api.py          FastAPI app, exposes `app`
tests/
```

The modules live in the `template/` package (one package per repo, named after the topic), run as
`python -m template.training`, with `config/` and `serving/` at the root. Root also holds
`Dockerfile.base` (builds `roshbeed/ai-residency-base`), `Dockerfile.train`, `Dockerfile.serve`,
`docker-compose.yml`, `Makefile`, `pyproject.toml`, `uv.lock` and this README. Trained artifacts go
to `checkpoints/` locally and to the Hugging Face Hub; neither `checkpoints/` nor `data/` is tracked
in git.

## The training shape

`train()` is the part worth copying exactly:

```
set_seed(config.seed)          # before anything touches the RNG
build_datasets / loaders
compute_baseline(...)          # what does the model have to beat?
init_tracking(config.wandb_project, ...)
for epoch in ...:
    ...
    log_metrics({"train_loss": ..., "val_loss": ...}, step=epoch)
    if val_loss < best_loss:
        save_checkpoint(...)                       # local
        push_checkpoint(config.hf_repo_id, ...)    # Hub, best only
finish_tracking()
```

Two of those are there for reasons this portfolio learned the hard way. Seeding comes first
because unseeded runs weren't just producing wobbly numbers, they were hiding real correctness
bugs — in two repos, adding a seed is what surfaced one. And a baseline is computed before
training because a loss with nothing to compare against says very little; in one repo the trained
model turned out not to beat predicting the mean, which the reported number alone never revealed.

The run ends by stating plainly whether it beat the baseline:

```
Best validation loss 5.2824 against baseline 12.9774 (beats baseline)
```

## Commands

```bash
make install       # uv sync
make train         # train, checkpoint, push best to the Hub
make serve         # FastAPI on :8000
make check         # lint + strict mypy + tests

make docker-build-base    # build roshbeed/ai-residency-base (needs GITHUB_TOKEN; only step that does)
make docker-build-train   # build the training image, FROM roshbeed/ai-residency-base
make docker-build-serve   # build the serving image, FROM roshbeed/ai-residency-base
make docker-up             # serve, in a container
make docker-train          # train, in a container (profile-gated)
```

`make help` lists everything, including the `docker-push-*` targets (need `docker login` first).

## Standards

Shared behaviour comes from [`ai-residency-common`](https://github.com/RoshBeed/ai-residency/tree/main/libs/common), pinned by tag.
Device selection, logging, checkpoint format, seeding, Hub upload and W&B tracking are defined
once there rather than six times here.

Ruff (format and lint) and mypy run in **strict mode with zero suppressions** — no `type: ignore`,
no blanket `noqa`. The `[tool.ruff]` and `[tool.mypy]` blocks are copied verbatim from
`ai-residency-common` so configuration cannot drift between repos.

Python is pinned at `>=3.11` with `python:3.11-slim` in every Dockerfile here, matching across the
portfolio.

## Docker: one base image, two purposes

`Dockerfile.base` installs `ai-residency-common` — the one private git dependency every repo shares — into
`roshbeed/ai-residency-base`. That's the only Dockerfile in the whole portfolio that ever needs a GitHub
credential. `Dockerfile.train` and `Dockerfile.serve` both build `FROM roshbeed/ai-residency-base` and never
touch git at all: `ai-residency-common` is already installed by the time their own `uv sync` runs, and `uv`
recognizes it as already-satisfied against the lock file rather than re-fetching it.

- **`Dockerfile.serve`** is what Kubernetes runs — the FastAPI inference container.
- **`Dockerfile.train`** is what a RunPod or AWS spot-GPU host pulls and runs for training.

Build the base image once locally before building either of the other two:
`make docker-build-base`. Once `roshbeed/ai-residency-base` is published, `docker-build-train`/
`docker-build-serve` can instead pull it (`--build-arg BASE_IMAGE=roshbeed/ai-residency-base:latest`).

## Credentials

Everything degrades rather than failing, so a fresh clone works without any of these:

| Variable | Absent |
|---|---|
| `HF_TOKEN` | Checkpoints stay local, warning logged |
| `WANDB_API_KEY` | Run goes offline, sync later with `wandb sync` |
| `GITHUB_TOKEN` | Only needed for `make docker-build-base` — `ai-residency-common` is private |

`docker-build-base` passes the token as a BuildKit secret rather than baking it in:
`docker build --secret id=gh_token,env=GITHUB_TOKEN`. The Makefile fills `GITHUB_TOKEN` from
`gh auth token` automatically.
