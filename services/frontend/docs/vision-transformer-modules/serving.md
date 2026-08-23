# `serving/` — the HTTP backend

`api.py` exposes `app`, a FastAPI service that loads both checkpoints on startup (from disk, or
pulled from the Hub when absent) and serves them. This is the canonical location for the backend
in every repo in this portfolio (`serving/api.py`), which is what lets the portfolio frontend talk
to all six the same way.

Run it with `make serve`, which is `python -m serving.api`.

## Endpoints

| Route | What it does |
| --- | --- |
| `GET /` | Health check: which models loaded, and the device they are on. |
| `GET /models` | Lists both models and their load status. |
| `POST /predict` | Predicts with whichever model `model_type` names (`vit` or `seq2seq`). |
| `POST /predict/vit` | The single-digit classifier specifically. |
| `POST /predict/seq2seq` | The multi-digit seq2seq model specifically. |
| `POST /predict/both` | Runs both over the same image and returns both answers. |

Request bodies carry the image base64-encoded in a pydantic field rather than as a multipart
upload, matching the convention the other week repos use:

```json
{ "image_data": "<base64 png>", "model_type": "vit" }
```

## One thing worth knowing

`preprocess_image` takes the target size as a required argument rather than defaulting it. The
two models take different resolutions (28×28 and 84×84), and sending a 28×28 image to the
seq2seq model does not raise: its patch convolution (kernel 16, stride 16) just reads one corner
and silently returns noise. Forcing the caller to state the size is what makes that mistake
impossible rather than invisible. `/predict/both` preprocesses the image twice for the same
reason.
