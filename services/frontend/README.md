# `services/frontend` — the Streamlit app

One page per system, each a thin HTTP client calling that system's own backend. **No inference
logic lives here** — only request/response wiring and display. The models run in their own
containers; this renders.

The portfolio overview lives in the [root README](../../README.md); this file is about the app.

```mermaid
flowchart LR
    UI["Streamlit frontend<br/>(this service)"]
    subgraph backends["One container per service"]
        B1["word-embeddings"]
        B2["content-scoring"]
        B3["vector-search-retrieval"]
        B4["vit-classifier"]
        B5["vit-seq2seq"]
        B6["multimodal-captioning"]
        B7["whisper-fine-tuning"]
        B8["speech-emotion-recognition"]
        B9["rlhf-ppo"]
    end
    UI -->|HTTP| B1 & B2 & B3 & B4 & B5 & B6 & B7 & B8 & B9
```

## What each system does

| System | Architecture | Demonstrated |
|---|---|---|
| [Word embeddings](https://github.com/RoshBeed/ai-residency/tree/main/services/word-embeddings) | CBOW word2vec with negative sampling, trained from scratch | Real unsupervised structure in the trained vectors (t-SNE clusters, analogy/similarity checks) |
| [Content scoring](https://github.com/RoshBeed/ai-residency/tree/main/services/content-scoring) | Regression head over frozen CBOW embeddings | Trained end-to-end on real Hacker News post data |
| [Dual-encoder retrieval](https://github.com/RoshBeed/ai-residency/tree/main/services/vector-search-retrieval) | Two-tower retrieval (CBOW + GRU encoders), in-batch negatives | Trained towers served via Redis/RediSearch ANN search |
| [ViT classifier](https://github.com/RoshBeed/ai-residency/tree/main/services/vit-classifier) | From-scratch encoder-only ViT over 4x4 patches | 91.78% on the MNIST test set against an 11.35% majority-class baseline |
| [ViT seq2seq](https://github.com/RoshBeed/ai-residency/tree/main/services/vit-seq2seq) | From-scratch encoder-decoder ViT, causal decoder and per-sequence EOS | 98.53% exact-sequence on one- to three-digit composites against 0.10% for a uniform guess |
| [Image captioning](https://github.com/RoshBeed/ai-residency/tree/main/services/multimodal-captioning) | CLIP vision encoder → adapter → Qwen3 decoder | Full pipeline trained and serving captions end-to-end |
| [Whisper fine-tuning](https://github.com/RoshBeed/ai-residency/tree/main/services/whisper-fine-tuning) | Whisper fine-tuned live, per request | Fine-tuning and data pipeline running end-to-end, served via FastAPI |
| [RLHF / PPO](https://github.com/RoshBeed/ai-residency/tree/main/services/rlhf-ppo) | From-scratch PPO — generation, KL penalty, reward, advantage, policy/value updates | Full PPO step runs end-to-end against a real preference-trained reward model |

## The write-ups

Each system's write-up — architecture, method, results, references — is that service's own
`README.md`, so it is both the GitHub landing page for the service and the document this app links
to. They used to live here, in a separate repo from the code they describe, because relative links
between repos 404 on GitHub.

| System | Write-up |
|---|---|
| Word embeddings (CBOW from scratch) | [`services/word-embeddings`](../word-embeddings) |
| Content scoring (regression head over frozen vectors) | [`services/content-scoring`](../content-scoring) |
| Dual-encoder document retrieval | [`services/vector-search-retrieval`](../vector-search-retrieval) |
| ViT classifier (single digit) | [`services/vit-classifier`](../vit-classifier) |
| ViT seq2seq (multi-digit) | [`services/vit-seq2seq`](../vit-seq2seq) |
| Image captioning (CLIP + Qwen3) | [`services/multimodal-captioning`](../multimodal-captioning) |
| Whisper fine-tuning | [`services/whisper-fine-tuning`](../whisper-fine-tuning) |
| Speech emotion recognition | [`services/speech-emotion-recognition`](../speech-emotion-recognition) |
| RLHF / PPO | [`services/rlhf-ppo`](../rlhf-ppo) |

The exploratory notebooks live at [`notebooks/`](../../notebooks), one folder per service.

## The app

One Streamlit app, one module per system under `views/`. Each page is a thin HTTP client calling
that system's own backend container — **no inference logic lives here**, only request/response
wiring and display. That's deliberate: the models run in their own services, and this repo just
renders.

Routing is `st.navigation`, so Streamlit owns the URL, the browser history and the sidebar.
`app.py` maps a service name to its `views/` module and gets out of the way; every page is an
ordinary function, which is what lets the suite render all nine of them headlessly with
`AppTest`.

![The portfolio home page: a card per system, each with its architecture summary and topic tags](assets/deployment/home.png)

Each page calls its backend live. Here the word-embeddings page returns real nearest neighbours,
with the cosine scores and the `checkpoint_source` the backend reported:

![The Word Embeddings page returning real nearest neighbours for a query word, with cosine scores](assets/deployment/word-embeddings-result.png)

## Run locally

```bash
make install          # uv sync
make run              # uv run streamlit run app.py
```

Each page reads its backend's URL from an environment variable named after the service —
`WORD_EMBEDDINGS_API_URL`, `VIT_CLASSIFIER_API_URL`, `SPEECH_EMOTION_RECOGNITION_API_URL` and so
on, defaulting to `http://localhost:8001`–`:8009`. Point them at whichever backends are up: the
compose stack, the Helm cluster via `kubectl port-forward`, or individual `make serve` processes.

The Helm chart derives these names from its `backends` keys, so adding a service needs no
frontend-side wiring beyond a page.

```bash
WORD_EMBEDDINGS_API_URL=http://localhost:8001 make -C services/frontend run
```

## Docker

```bash
# from the repository root
docker build -f services/frontend/Dockerfile.serve -t roshb/ai-residency-frontend:latest .
docker push roshb/ai-residency-frontend:latest
```

It does not build `FROM ai-residency-base-serve`, deliberately — that image carries a torch install this
one has no use for.

The image is deliberately lightweight — `streamlit` + `requests` + `pillow` only. Every model runs in
its own backend container.

## Layout

```
services/frontend/
├── app.py            entry point: page config, theme, st.navigation
├── views/            one module per page, each exporting render()
│   ├── home.py         the gallery — a card per system
│   └── <service>.py    that system's demo and the trace it renders
├── shared.py         the framing every service page shares, and the small helpers
├── client.py         one call() returning ok / not_ready / failed
├── components.py     pure display helpers — no Streamlit import
├── projects.py       per-system metadata: titles, tags, diagrams, links
├── theme.py          logo and CSS
├── assets/           figures the pages render inline
├── Dockerfile.serve  the published roshb/ai-residency-frontend image
├── Makefile          run
└── pyproject.toml    streamlit + requests + pillow — no ML dependencies
```

## Related

- [`deploy`](https://github.com/RoshBeed/ai-residency/tree/main/deploy) — compose, Helm, Terraform to run all of it.
- [`libs/common`](https://github.com/RoshBeed/ai-residency/tree/main/libs/common) — the shared helper library all the backends depend on.
- The backend repos linked in the table above.

---

*These systems originated as projects in the [Machine Learning Institute](https://ml.institute/) residency.*
