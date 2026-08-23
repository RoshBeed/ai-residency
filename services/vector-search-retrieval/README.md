# Two-Tower Dual-Encoder Document Retrieval

## TL;DR

- **What it is:** a dual-encoder retrieval system — a query tower and a document tower trained end-to-end with triplet loss, with every passage embedded once and indexed for approximate nearest-neighbour search.
- **Built:** the full offline/online split — CBOW embeddings, triplet-loss training with a W&B sweep, a held-out evaluation harness, and Redis/RediSearch HNSW indexing for live queries.
- **What I'd do next:** train at real MS MARCO scale, report Recall@K/MRR against a BM25 baseline, add hard-negative mining.

This repo builds a neural document retrieval system: given a search query, find the most relevant
passage in a corpus. I trained a query tower and a document tower with a triplet loss on real
search queries (MS MARCO), so a query's vector lands close to its relevant passage and far from
irrelevant ones. Once trained, every passage gets embedded once and indexed for approximate
nearest-neighbour search, so retrieval at query time is a single vector lookup instead of scoring
every passage in the corpus. This is the dual-encoder architecture behind modern semantic/vector
search.

## Quickstart

```bash
# one venv for the whole workspace
make install

# FastAPI on :8000, interactive docs at /docs
make serve SERVICE=vector-search-retrieval

# read Usage first -- some stages run for hours
make train SERVICE=vector-search-retrieval
```

Stages specific to this service live in its own Makefile (`make -C services/vector-search-retrieval help`):

- `train-cbow` — Build the vocabulary, then train CBOW word embeddings
- `train-triples` — Build and tokenize MS MARCO triples
- `train-encoder` — Train the GRU dual-encoder towers
- `index` — Embed the corpus with the trained doc tower and index it in Redis
- `query` — Index the corpus, then the interactive query CLI (needs Redis running)
- `redis` — Start a throwaway RediSearch-capable Redis on :6390

Checkpoints are resolved locally first, then from the Hugging Face Hub, so serving works on a host that has never trained anything.

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Method](#3-method)
4. [Implementation](#4-implementation)
5. [Usage](#5-usage)
6. [Experimental Results](#6-experimental-results)
7. [Related Work](#7-related-work)
8. [References](#references)
9. [Acknowledgments](#acknowledgments)
10. [Dependencies](#dependencies)

## 1. Introduction

Every search system is, at its core, one function: given a document and a query, score how similar
or relevant they are.

**Figure 1.** The general shape of a ranking problem: a document and a query both feed a function
that scores how similar or relevant they are.
![The general shape of a ranking problem: document and query both feed a "magic function" that outputs how similar or relevant they are.](assets/fig1-magic-function.png)

Every technique in this repo is one way of instantiating that "magic function": TF-IDF makes it a
term-overlap statistic, a dual-encoder makes it a learned similarity between two vectors.

Classical information retrieval answers this with hand-built statistics rather than a learned
function: TF-IDF scores a document by how often a query term appears in it, discounted by how
common that term is across the whole corpus, and an inverted index makes looking up "which
documents contain this term" fast at scale. That's genuinely how search engines worked for
decades (and still partly do), but it only matches on literal term overlap: a query for "was
ronald reagan a democrat" won't match a passage that answers the question without using those
exact words. This repo builds the learned alternative instead: encode both query and document as
vectors and score them by similarity, so a match is about meaning, not shared vocabulary.

There are two ways to build that learned function. A **cross-encoder** feeds query and document
tokens into one joint model with full cross-attention between them. That's accurate, but every
candidate document has to be re-encoded together with the query at query time, so it doesn't scale
to a large corpus. A **dual-encoder** ("two-tower") splits query and document into two independent
encoders that never see each other during inference, so every document can be embedded once,
offline, and reused for every future query.

**Figure 2.** Dual-encoder versus cross-encoder. Dual-encoder query and document tokens only
interact after each side is fully encoded (right at the SCORE node); the cross-encoder lets every
query token attend to every document token before scoring.
![Dual-Encoder architecture: separate QUERY and DOC encoder stacks meeting only at a final SCORE node. Cross-Encoder architecture: QUERY and DOC tokens feed one joint encoder stack with cross-attention before SCORE.](assets/fig2-dual-vs-cross-encoder.png)

This repo builds the dual-encoder path: `training.py` implements the
left-hand diagram, a separate GRU stack for QUERY and DOC that never cross-attend.

## 2. System Architecture

Before the method — here's how the whole thing is wired together at a glance. Five stages, split
into an offline half (train embeddings and towers, index the corpus) and an online half (embed and
answer a live query):

**Figure 3.** Full pipeline: embeddings, triples, training, and indexing, split into an offline and
an online half. Left, offline: CBOW word2vec pretraining, MS MARCO triples through the query/doc
towers into the contrastive loss, and the trained doc tower indexing the corpus into Redis. Right,
online: the same query path, but stopping at the tower and going into an ANN search against that
Redis index to return the top-5 passages.
![Left panel, offline: text8 and MS-MARCO feed Train word2vec and a CBOW tokeniser respectively; Query Token and Doc Token pass through a shared Embedding Layer and Average Pooling into a Query Tower and Doc Tower; Query Embedding and Doc Embedding feed a Contrastive Loss node, and Doc Embedding also writes into a Redis Database. Right panel, online: the same text8/MS-MARCO/tokeniser path plus a User Query actor feed the Query Tower; its Query Embedding feeds an ANN Search against the Redis Database, returning the top 5 most relevant documents.](assets/fig7-pipeline-evolution.png)

This is the same pipeline `src/` implements, just with a couple of naming/store differences worth
calling out: the diagram's "Doc Tower"/"Query Tower" boxes are `models/towers.py`'s `GruTower` —
one class, two separately-trained instances, sharing only the frozen embedding layer beneath them —
the "Tokenise with CBOW Tokeniser" + Query/Doc Token split is `data_prep/generate_triples.py` +
`tokenize_triples.py`, and the Redis Database node is written by
`retrieval/save_doc_embeddings_to_redis.py` and read by `retrieval/query_engine.py`, which loads
the trained query tower and is shared by the CLI, the FastAPI service, and the Streamlit UI.

Same pipeline, as a sequence diagram: one node per script, in the order they're actually run.

**Figure 4.** Sequence diagram of the pipeline.
```mermaid
sequenceDiagram
    actor dev
    participant gen as generate_triples.py
    participant tok as tokenize_triples.py
    participant train as training.py
    participant index as save_doc_embeddings_to_redis.py
    participant redis as Redis (RediSearch)
    participant query as query_redis_ann.py

    dev->>+gen: python -m data_prep.generate_triples
    gen-->>-dev: data/triples_small.json

    dev->>+tok: python -m data_prep.tokenize_triples
    tok-->>-dev: data/tokenized_triples.json

    dev->>+train: python -m training
    train-->>-dev: trained query/doc tower weights

    dev->>+index: python -m retrieval.save_doc_embeddings_to_redis
    index->>redis: FT.CREATE index, HSET per passage vector
    deactivate index

    dev->>+query: python -m retrieval.query_redis_ann
    query->>query: embed query (query tower)
    query->>+redis: FT.SEARCH ... KNN
    redis-->>-query: top-K passage ids + scores
    query-->>-dev: ranked results
```

## 3. Method

### 3.1 Encoding Text as a Vector

An encoder, in this sense, is just a function that turns any piece of text into a fixed-size
vector (the same idea covered in depth in the
[word-embeddings/score-prediction project](https://github.com/RoshBeed/ai-residency/tree/main/services/word-embeddings)): tokenize the text, look
up a (frozen, pretrained) word embedding for each token, and pool them into one vector, e.g. by
averaging. Given any sentence, the model produces a vector representation of it; the rest of this
project is about learning *which* vector representation is useful for retrieval specifically.

**Figure 5.** Query and document towers at inference time: a query is looked up in an embedding
table, averaged into a vector, and fed into a QUERY tower; candidate documents are fed into a DOC
tower; both towers converge on a single SCORE.
![A query "was ronald reagan a democrat" is looked up in an embedding table, averaged into a 1x5 vector, and fed into a QUERY tower; two candidate documents are fed into a DOC tower; both towers converge on a single SCORE.](assets/fig3-two-tower-search-systems.png)

The diagram's "average, then feed a tower" step is `data_prep/create_triple_embeddings.py` for
the mean-pooling variant, and the tower boxes themselves are
`models/towers.py`'s `GruTower` for the trained variant
actually used for indexing — same shape, learned recurrence instead of a plain average.

Both towers here sit on top of word embeddings trained from scratch with a small CBOW model (see
`cbow/`, the same recipe as the word-embeddings project), frozen and shared between the two towers. I tried two
tower architectures on top of those frozen embeddings:

- **Average-pooling towers** (`create_triple_embeddings.py`) — mean the frozen CBOW word vectors
  over a document/query, no additional learned parameters.
- **GRU towers** (`models/towers.py`, `models/training.py`, `save_doc_embeddings_to_redis.py`,
  `query_redis_ann.py`) — a learned recurrent encoder on top of the frozen embeddings, so word
  order (not just word identity) affects the resulting vector. This is the version actually
  trained with a hyperparameter sweep and used for indexing.

### 3.2 Training Data: MS MARCO

Training needs examples of "this document answers this query" and "this document doesn't."
**MS MARCO v1.1** provides real search queries, a set of candidate passages per query, and a
binary relevance label per passage — everything needed to build a training triple.

**Figure 6.** Turning an MS MARCO example into a training triple: a real screenshot of the actual
v1.1 Hugging Face dataset viewer, with the actual `is_selected` and `passage_text` fields this
repo's own loader reads.
![The MS MARCO v1.1 Dataset Viewer, showing a row with is_selected: [0, 1, 0, 0, 0, 0, 0], passage_text: ["A".."G"], and url: [...]. An arrow points to a JSON snippet marking is_selected==1 entries as "positive" and an is_selected==0 entry as "maybe hard negative".](assets/fig4-ms-marco-triple.png)

`generate_triples.py` builds `(query, positive_doc, negative_doc)` triples this way, and
`tokenize_triples.py` converts each into token ids using the CBOW vocabulary.

### 3.3 Triplet Loss

With a triple in hand, the training signal is simple: push the query embedding toward its positive
document and away from its negative one, by at least a margin $m$.

**Figure 7.** Triplet training signal: the margin geometry (left) and the score/loss wiring (right)
that this repo's own margin (`0.2`, see the code below) and towers implement.
![Left: a circle diagram showing Query, Positive Doc, and Negative Doc points, with d+ = 1.05 inside a margin: 0.65 ring and d- = 2.01 outside it. Right: POSITIVE DOC, QUERY, and NEGATIVE DOC tower stacks feeding two SCORE nodes, which feed the contrastive loss formula max(0, m - s_cosine(q, d+) + s_cosine(q, d-)).](assets/fig5-triplet-loss.png)

$$
\mathcal{L}_{\text{contrastive}} = \max\big(0,\ m - s_{\cos}(q, d^{+}) + s_{\cos}(q, d^{-})\big)
$$

If the positive is already more similar than the negative by at least the margin, the loss is
zero and there's nothing to correct; otherwise the towers get pushed to widen that gap. The
smallest possible version of this is only a few lines:

```python
qryTower = QryTower()
docTower = DocTower()

qry = qryTower(query_embedding)
pos = docTower(positive_doc_embedding)
neg = docTower(negative_doc_embedding)

dist_pos = torch.nn.functional.cosine_similarity(qry, pos)
dist_neg = torch.nn.functional.cosine_similarity(qry, neg)
margin = torch.tensor(0.2)

loss = torch.max(torch.tensor(0.0), margin - (dist_pos - dist_neg))
loss.backward()
```

`models/training.py` is this same idea with GRU towers instead of a linear layer, trained
with a Weights & Biases sweep over learning rate, hidden size, and margin.

### 3.4 Indexing and Approximate Nearest-Neighbour Search

Training happens once, offline. After that, the document tower embeds the entire corpus once and
those vectors go into a vector index; at query time, only the query needs to be embedded, and the
index does the work of finding its nearest neighbours.

**Figure 8.** From trained towers to a queryable index, in three stages: train both towers
offline, embed the corpus offline through the doc tower, then embed and search only the query
online.
![Stage 1, Offline: train both query and document tower together via a SCORE node. Stage 2, Offline: a corpus of passages goes through the DOC TOWER into passage vectors, indexed in chromadb. Stage 3, Online: the query "was ronald reagan a democrat" goes through the QRY TOWER into a query vector, which does Approximate Nearest Neighbor search against chromadb.](assets/fig6-offline-online-stages.png)

The reference stack for this project uses ChromaDB for the vector index. I indexed in
**Redis** instead, using the [RediSearch](https://redis.io/docs/stack/search/) module's HNSW
(Hierarchical Navigable Small World) index for approximate nearest-neighbour search — a different
vector store, same role in the pipeline.

## 4. Implementation

Mapping the architecture from [Section 2](#2-system-architecture) onto the actual file layout:

```
src/
├── cbow/                       Standalone CBOW word-embedding trainer (reused from the word-embeddings project)
│   ├── train_tokenizer.py        Build tokenizer/vocab from raw corpus
│   ├── train_word2vec.py         Train CBOW embeddings
│   ├── train_regressor.py        (legacy regressor, not used by the retrieval pipeline)
│   ├── model.py / dataset.py      CBOW model + Wiki dataset
│   └── checkpoints/               Trained embedding weights
│
├── data_prep/                  MS MARCO -> tokenized training triples
│   ├── extract_data.py           (optional) EDA — dump MS MARCO queries/passages to JSON
│   ├── generate_triples.py       Build (query, positive_doc, negative_doc) triples from MS MARCO
│   ├── tokenize_triples.py       Tokenize triples using the CBOW vocabulary
│   ├── load_dataset.py           Standalone script to inspect MS MARCO dataset stats
│   ├── create_embeddings.py      Early experiment: average-pool triples with a randomly-init embedding
│   ├── create_triple_embeddings.py  Average-pool triples with the trained CBOW embeddings, push to Redis
│   └── upload_to_hf.py / download_from_hf.py  Push/pull checkpoints, embeddings and data
│                                    to/from Hugging Face Hub so training can resume elsewhere
│
├── models/                     Dual-encoder tower architectures + training loops
│   ├── dual_encoder.py            First dual-encoder draft (mean-pooled towers + triplet loss)
│   ├── towers.py                  GruTower — the GRU-based query/doc tower architecture
│   └── training.py                Trains the GRU towers, with a W&B sweep (config/sweep.yaml)
│
└── retrieval/                  Indexing + querying against the trained towers
    ├── query_engine.py               Shared core: load tower/vocab, embed a query, run the Redis
    │                                  KNN search — imported by both entry points below
    ├── save_doc_embeddings_to_redis.py  Embed the full document corpus with the trained doc
    │                                     tower and index it in Redis for ANN search
    └── query_redis_ann.py            Interactive CLI on top of query_engine.py

checkpoints/                  Trained query/doc GRU tower weights (qry_tower.pth, doc_tower.pth)
data/                         Generated training data (triples_small.json, tokenized_triples.json)
config/sweep.yaml             W&B hyperparameter sweep config for models/training.py
notebooks/walkthrough.ipynb   Narrates all five pipeline steps with real, importable calls
scripts/script.sh             Reference pipeline: extract → generate triples → tokenize → train
```

## 5. Usage

Requires a running Redis instance with the [RediSearch](https://redis.io/docs/stack/search/)
module (e.g. Redis Cloud or `redis-stack` locally) for the vector index, and `REDIS_HOST` /
`REDIS_PORT` / `REDIS_PASSWORD` in your environment (or a `.env` file).

### Docker Compose (quickest)

```bash
make -C deploy up      # starts Redis + the UI, builds+runs the serve image: index corpus, then interactive query CLI
docker compose --profile train run --rm train-vector-search-retrieval
make build SERVICE=vector-search-retrieval    # build the serve, ui, and train images without running them
make -C deploy down     # stop and remove all Compose services
```

`docker-up` starts a `redis-stack-server` container (RediSearch module, no persistence, throwaway
by design) and then runs the `serve` image, which is this repo's stand-in for a "serve" step:
there's no persistent HTTP server here, so it indexes the corpus and drops into the same
interactive `query_redis_ann.py` CLI described below. `docker-train` runs the offline pipeline
(CBOW embeddings -> triples -> tower training) inside the `train` image, one stage at a time or
all five in order.

### Local (Makefile)

```bash
make install   # uv sync
make train SERVICE=vector-search-retrieval     # train-cbow && train-triples && train-encoder, in order
make serve SERVICE=vector-search-retrieval     # the FastAPI query service (needs Redis already running)
make test SERVICE=vector-search-retrieval      # pytest
```

The interactive query CLI is `python -m retrieval.query_redis_ann` (a local-only loop, not a
server). The browser UI is the unified Streamlit frontend in `services/frontend`, which calls
this repo's `serving/api.py` over HTTP — it is not in this repo.

Local `make train SERVICE=vector-search-retrieval*` is the fast path on Apple Silicon: CBOW training picks up the MPS
backend automatically when available, benchmarked ~3.7x faster than CPU on this repo's own
batch/vocab shape. `make -C deploy up  # then: docker compose --profile train run --rm train-vector-search-retrieval` always runs CPU-only, even on this same Mac (Docker
Desktop's containers run inside a Linux VM with no GPU/MPS passthrough), so it's for portability
and eventual real-GPU-host deployment, not local speed. (The GRU dual-encoder training in
`models/training.py` trains one triple at a time on tiny tensors; benchmarked, MPS is
actually slower there due to per-op dispatch overhead, so it intentionally stays CPU-only.)

### Raw commands (reference)

What the targets above actually run, if you'd rather call the scripts directly:

```bash
uv sync

# 1. Train word embeddings (or download pretrained ones — see data_prep/download_from_hf.py)
uv run python -m cbow.train_tokenizer
uv run python -m cbow.train_word2vec

# 2. Build training triples from MS MARCO
uv run python -m data_prep.generate_triples
uv run python -m data_prep.tokenize_triples

# 3. Train the dual encoder (GRU towers, with a W&B sweep)
uv run python -m training

# 4. Index the document corpus in Redis
uv run python -m retrieval.save_doc_embeddings_to_redis

# 5. Query it interactively (CLI), or call the FastAPI service
uv run python -m retrieval.query_redis_ann
uv run python -m serving.api
```

`scripts/script.sh` documents the same sequence end-to-end. `notebooks/walkthrough.ipynb` narrates
all five steps with real, importable calls into these same scripts, cell by cell.

## 6. Experimental Results

The full pipeline runs end to end — CBOW embeddings, triplet-loss tower training, Redis indexing,
live query — reproduced here on a reduced scale (300k-token corpus, 444 training triples from real
MS MARCO examples) with a fixed seed (`torch.manual_seed(42)`) for reproducibility.

Triplet loss on the training triples dropped from 0.2026 (epoch 1) to 0.0032 (epoch 10). On the
training queries, positive-document cosine similarity was consistently well above negative-document
similarity:

| Query | sim(query, positive doc) | sim(query, negative doc) |
|---|---|---|
| "what is rba" | 0.2801 | −0.4856 |
| "was ronald reagan a democrat" | 0.3507 | −0.2441 |
| "price to install tile in shower" | 0.3726 | −0.1499 |

A held-out evaluation harness (`pairwise_accuracy()` in `models/training.py`) checks generalization
directly — whether the query lands closer to its positive document than its negative one, on
triples the training loop never saw — rather than trusting the training loss alone.

A live query against the full indexed corpus (1,498 documents in Redis), through the trained query
tower and Redis's HNSW search:

```
Enter your query: was ronald reagan a democrat
Top 5 results:
Rank 1 (score=0.5999): "[ 2 syll. kei (l), ke-il ] The baby boy name Keil is also used as a girl
  name. Its pronunciation is..."
Rank 2 (score=0.6432): "You should consult the doctor immediately. In the abdomen many organs are
  covered and kept in a bag..."
Rank 3 (score=0.6476): "CUPRO \¦k(y)ueˈprō\. üˈprō. n 1: A a regenerated cellulose fiber derived
  from cotton (linter the, ul..."
Rank 4 (score=0.6757): "System Costs. As mentioned in the introduction, residential solar systems
  are typically sized from 3..."
Rank 5 (score=0.6838): "Home » Tramadol Uses, Warnings and Treatment. Tramadol is a
  narcotic-like pain reliever. Tramadol hy..."
```

That transcript is real — captured live from `run_query()` in `retrieval/query_engine.py` against
the trained query tower and the actual indexed corpus.

## 7. Related Work

- [The word-embeddings/score-prediction project](https://github.com/RoshBeed/ai-residency/tree/main/services/word-embeddings) — the
  CBOW embedding training reused here as the frozen input representation for both towers.
- [The vision transformer project](https://github.com/RoshBeed/ai-residency/tree/main/services/vision-transformers) — self-attention
  implemented from scratch, a natural upgrade path for the tower encoders in place of a GRU.

## References

[1] Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., & Shah, R. (1994). *Signature Verification
using a "Siamese" Time Delay Neural Network.* NeurIPS. — the original twin-network architecture
that dual-encoders are a direct descendant of.

[2] Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020).
*Dense Passage Retrieval for Open-Domain Question Answering.* arXiv:2004.04906.

[3] Brin, S., & Page, L. (1998). *The Anatomy of a Large-Scale Hypertextual Web Search Engine.*
Computer Networks and ISDN Systems, 30(1–7). — classical, non-neural context for the ranking
problem this project addresses with a learned model instead.

## Acknowledgments

Built with Ewan Beattie, James Carter, and Melanie Wong.

## Dependencies

PyTorch · MS MARCO (Hugging Face `datasets`) · Redis / RediSearch (HNSW ANN index) ·
Weights & Biases (sweeps) · Hugging Face Hub (artifact storage)
