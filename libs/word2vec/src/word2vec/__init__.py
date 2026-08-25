"""word2vec from scratch: CBOW and skip-gram, negative sampling or full softmax.

Three services in this repo grew their own word2vec, one per team that worked on it,
and they disagreed about everything except the maths. This is the one implementation
they now share. `model.py` is the file to read to understand the algorithm.

    from word2vec import NegativeSampler, Word2Vec, Word2VecConfig

    config = Word2VecConfig(vocab_size=50_000, embed_size=128, architecture="cbow")
    model = Word2Vec(config)
    sampler = NegativeSampler(corpus_counts)

    negatives = sampler(len(centre), config.num_negatives)
    loss = model(centre, context, negatives)   # a loss, whatever the variant
    loss.backward()

    vectors = model.embeddings                 # what everything downstream wants

Checkpoints written by the older implementations still load; see `compat.py`.
"""

from word2vec.compat import LEGACY_KEY_MAPS, state_dict_from_legacy
from word2vec.config import Word2VecConfig
from word2vec.model import Word2Vec
from word2vec.model_types import Architecture, Objective
from word2vec.neighbours import nearest_words, valid_row_mask
from word2vec.sampling import NegativeSampler, unigram_distribution

__all__ = [
    "LEGACY_KEY_MAPS",
    "Architecture",
    "NegativeSampler",
    "Objective",
    "Word2Vec",
    "Word2VecConfig",
    "nearest_words",
    "state_dict_from_legacy",
    "unigram_distribution",
    "valid_row_mask",
]
