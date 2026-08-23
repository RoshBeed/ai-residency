# `serving/` — the HTTP backend

`api.py` exposes `app`, a FastAPI service that loads the multi-digit encoder-decoder model on startup — from disk, or
pulled from the Hugging Face Hub when absent — and serves it. `serving/api.py` is the canonical
location for the backend in every service in this portfolio, which is what lets the frontend talk
to all of them the same way.

Routes: `/predict`, `/predict/stream` (per-digit SSE), plus `/health` and `/models`.

Images arrive base64-encoded in a pydantic field rather than as multipart uploads, matching the
convention every other service uses.
