# word2vec

CBOW and skip-gram, with a negative-sampling or full-softmax objective. Written from
scratch — the maths is in [`model.py`](src/word2vec/model.py), which is the file to read.

Three services here grew their own word2vec, one per team that worked on it, and they
agreed about nothing except the algorithm. This is the one implementation they share.

## The two axes

They are independent, which is why this is one module rather than four. The architecture
decides *which* vectors get combined; the objective decides *how* a score becomes a loss.

| architecture | |
|---|---|
| `cbow` | average the context vectors, predict the centre word |
| `skipgram` | take the centre vector, predict each context word. Slower per example, better on rare words |

| objective | |
|---|---|
| `negative_sampling` | score the target against `num_negatives` sampled words. O(num_negatives) |
| `softmax` | score against the whole vocabulary. Exact, and O(vocab_size) — which is what negative sampling exists to avoid |

```python
from word2vec import NegativeSampler, Word2Vec, Word2VecConfig

config = Word2VecConfig(vocab_size=50_000, embed_size=128, architecture="cbow")
model = Word2Vec(config)
sampler = NegativeSampler(corpus_counts)

negatives = sampler(len(centre), config.num_negatives)
loss = model(centre, context, negatives)  # a loss, whatever the variant
loss.backward()

vectors = model.embeddings  # (vocab_size, embed_size)
```

`forward` returns a **loss** rather than logits for either objective. That is the point:
a training loop written against this does not change when the variant does.

## Two matrices, always

```
in_embed   (vocab_size, embed_size)   the vectors you keep
out_embed  (vocab_size, embed_size)   the vectors you discard after training
```

A full-softmax implementation usually writes the second as `nn.Linear(embed_size,
vocab_size, bias=False)`, whose weight is *also* `(vocab_size, embed_size)`. Same matrix,
different name — which is what made unifying the old implementations a rename rather than
a retrain. [`compat.py`](src/word2vec/compat.py) does that mapping, and is meant to be
deleted once every service has been retrained under this module.

## One thing that surprises people

`out_embed` is initialised to zeros, as the reference implementation does. Gradient
reaches `in_embed` only *through* `out_embed`, so the first optimiser step moves only
`out_embed` — the vectors you actually keep get exactly zero gradient on step one.

A loss that drops on the first step therefore says nothing about the embeddings yet. The
test suite pins this down so it cannot be mistaken for a bug later.

## Sampling

Negatives are drawn from the unigram distribution raised to the 3/4 power, per Mikolov et
al. (2013). Uniform sampling wastes almost every negative on a rare word the model already
scores near zero; raw frequencies spend almost all of them on `the`. The exponent sits
between the two, and `unigram_distribution(counts, power=...)` lets you see that directly.

Words with a zero count are never sampled, so padding and unused ids cannot turn up as
negatives.

## References

Mikolov et al., [Efficient Estimation of Word Representations in Vector
Space](https://arxiv.org/abs/1301.3781) (2013) — CBOW and skip-gram.

Mikolov et al., [Distributed Representations of Words and Phrases and their
Compositionality](https://arxiv.org/abs/1310.4546) (2013) — negative sampling, and the 3/4
power.
