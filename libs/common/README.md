# ai-residency-common

Shared helpers for the six repos of my [Machine Learning Institute](https://ml.institute/)
residency portfolio. Each week is its own standalone repo, so before this package existed the
same concerns were implemented six separate times and drifted apart: device selection had six
implementations (one omitted CUDA entirely, one returned bare strings instead of `torch.device`,
one repeated the same block in three files), there were ~495 `print()` calls and no logging
config, checkpoint files had no agreed shape, four of the six repos had no random seed at all,
only two logged metrics anywhere, and only one pushed weights off the machine that trained them.

This package is small on purpose. It holds the things that must be identical across repos, and
nothing else — the actual ML lives in the week repos.

## Install

```bash
uv add "ai-residency-common @ git+https://github.com/RoshBeed/ai-residency-common.git@v0.7.3"
```

Pin by tag. Six repos depend on this, and an unpinned change would break all of them at once.

The repo is private, so `docker build` needs a token passed as a BuildKit secret rather than
inheriting host credentials:

```dockerfile
RUN --mount=type=secret,id=gh_token \
    git config --global credential.helper 'store --file=/tmp/gc' && \
    printf 'https://x-access-token:%s@github.com\n' "$(cat /run/secrets/gh_token)" > /tmp/gc && \
    uv sync --frozen --no-install-project && \
    rm -f /tmp/gc && git config --global --unset credential.helper
```

Built with `docker build --secret id=gh_token,env=GITHUB_TOKEN`. The token stays out of the image:
no credential file survives the layer, and it appears in no layer's contents.

## What's in it

```python
from ai_residency_common import (
    BaseConfig,
    decode_audio,
    decode_image,
    finish_tracking,
    format_sse,
    get_device,
    get_logger,
    init_tracking,
    load_checkpoint,
    load_env,
    load_model,
    log_metrics,
    pull_checkpoint,
    push_checkpoint,
    resolve_checkpoint,
    resolve_required,
    run_app,
    save_checkpoint,
    set_seed,
    stream_sse,
)
```

**`get_device()`** returns a `torch.device`, preferring CUDA, then MPS, then CPU. On an Apple
Silicon host that's `mps`; inside the Linux containers this portfolio deploys to it's `cpu`, since
Docker Desktop exposes neither accelerator to its VM.

**`get_logger(__name__)`** returns a stdout logger with a consistent format. Level comes from
`LOG_LEVEL` (default `INFO`), so verbosity is adjustable from a container or Makefile target
without touching code.

**`save_checkpoint()` / `load_checkpoint()`** read and write one checkpoint shape
(`model_state_dict`, `optimizer_state_dict`, `epoch`, `metrics`, `extra`). `load_checkpoint`
defaults to `map_location="cpu"`, which is not cosmetic: weights trained on MPS carry `mps:0`
tensor metadata and fail to deserialise in a CPU-only container.

**`BaseConfig`** is the dataclass each repo's own `Config` inherits, so `seed`, `batch_size`,
`learning_rate`, `num_epochs`, `checkpoint_dir`, `data_dir`, `hf_repo_id` and `wandb_project` are
named and typed the same everywhere. **`set_seed()`** seeds Python and PyTorch RNGs together.
Seeding earned its place here — adding it to repos that lacked it didn't just make runs
reproducible, in two cases it exposed a real correctness bug that run-to-run variance had been
hiding.

Any config field can be overridden from the environment as `AI_RESIDENCY_<FIELD_NAME>` in upper case:

```bash
AI_RESIDENCY_NUM_EPOCHS=1 make train        # a smoke run
AI_RESIDENCY_NUM_EPOCHS=50 make train       # the real thing, on a rented GPU box
```

The override walks the dataclass fields rather than a fixed list, so a repo's own hyperparameters
(`AI_RESIDENCY_NUM_SAMPLES`, `AI_RESIDENCY_SAMPLES_PER_CLASS`) work without that repo opting in. Values are converted
to the field's declared type and a value that doesn't convert raises rather than being ignored — a
typo'd override that silently trained on the default would be worse than a crash. Sequences are
comma-separated; an empty value clears an optional field, which is how a run opts out of pushing
(`AI_RESIDENCY_HF_REPO_ID=`).

**`load_env()`** loads `.env` files, and is called once when the package is imported, so
`HF_TOKEN` and `WANDB_API_KEY` reach the functions below without being exported by hand. Every
`.env` from the working directory up to the root is loaded, nearest first: one file beside the
repos covers all of them, and a repo needing something of its own adds a `.env` with just that.

Stopping at the first file found is the obvious implementation and is wrong — one repo here has a
Redis-only `.env`, which silently shadowed the shared credentials one directory up, so that repo
trained without a token while the other five did not. Layering them fixes it. Nothing already set
in the environment is overridden, which is what makes this safe in containers, where the values
come from compose's `environment:` block or a Kubernetes Secret and no `.env` exists at all.

**`push_checkpoint()` / `pull_checkpoint()`** host trained weights on the Hugging Face Hub.
Checkpoints are gitignored in every repo, so a fresh clone or a published image pulled from Docker
Hub contains no usable model; the infrastructure repo previously worked around that by
bind-mounting checkpoint paths from sibling repo directories on one specific machine, which is
precisely why it could not be deployed anywhere else. Push only when a new best checkpoint is
written — these artifacts run to ~1.2GB and re-uploading every improvement during a sweep would
dominate training time.

All three take `repo_type="dataset"` as well, for training *inputs* that cannot simply be fetched
again — our own extracts and derived corpora. Third-party datasets do not belong there: they have
their own homes and their own licences, and re-hosting them is a legal question rather than a
technical one.

**`resolve_checkpoint(path, hf_repo_id)`** is the serving half of that, and is what lets a service
run on a host that never trained anything: it returns the local file if there is one, otherwise
downloads it, and reports which happened as `"local"` / `"hub"` / `"none"`. Every backend surfaces
that from `/health`, because a service that quietly fell back to an untrained model still answers
requests — just with nonsense — and that is invisible from the outside otherwise.

**`init_tracking()` / `log_metrics()` / `finish_tracking()`** wrap Weights & Biases so every repo
reports metrics the same way, rather than printing losses to stdout where they can't be compared
across runs.

The serving helpers (`sse.py` / `serving.py`) are the pieces every FastAPI backend used to write
itself: **`format_sse`** / **`stream_sse`** turn a worker into a `text/event-stream` response,
**`resolve_required`** is `resolve_checkpoint` plus a hard `FileNotFoundError`, **`load_model`**
resolves + loads + evals a torch model, **`run_app`** is the uvicorn bootstrap every `make serve`
ends in, and **`decode_image`** / **`decode_audio`** decode the base64 binary payloads the backends
accept.

Both degrade rather than fail. With no Hugging Face token, training runs and saves locally after
logging a warning. With no `WANDB_API_KEY`, the run goes to offline mode and can be synced later
with `wandb sync`. Someone cloning a week repo without these accounts can still run `make train`.

## Development

```bash
make install
make check     # lint + strict mypy + tests
```

Ruff (format and lint) and mypy run in strict mode with zero suppressions. The `[tool.ruff]` and
`[tool.mypy]` blocks in `pyproject.toml` are the canonical copies for the whole portfolio — the
week repos use the same configuration.
