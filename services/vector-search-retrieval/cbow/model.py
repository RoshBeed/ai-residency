"""The CBOW model this service trains its vocabulary with.

The algorithm lives in `libs/word2vec` now, shared with word-embeddings, which had
its own copy. The two were not rival CBOWs -- they were the same two matrices under
different objectives. This one used a full softmax over the vocabulary and called the
matrices `emb` and `ffw`; the other used negative sampling and called them `in_embed`
and `out_embed`. `ffw` is `nn.Linear(embed, vocab, bias=False)`, whose weight is the
same `(vocab, embed)` shape as an output embedding, so the difference was naming.

Only `in_embed` matters downstream: the retrieval towers want the embedding matrix and
discard the rest.
"""

from word2vec import Word2Vec, Word2VecConfig

__all__ = ["build_cbow"]


def build_cbow(vocab_size: int, embedding_dim: int) -> Word2Vec:
    """Build the CBOW model this service trains.

    Full softmax rather than negative sampling, which is what this pipeline has always
    used -- exact, and affordable at this vocabulary size.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        embedding_dim: Width of each word vector.

    Returns:
        An untrained model. `model.embeddings` is the matrix the towers consume.
    """
    return Word2Vec(
        Word2VecConfig(
            vocab_size=vocab_size,
            embed_size=embedding_dim,
            architecture="cbow",
            objective="softmax",
        )
    )
