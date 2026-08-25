"""CBOW word embeddings.

The towers in this repo do not learn word meaning from scratch; they sit on frozen
embeddings trained here first, on `text8`. `train_tokenizer` builds the vocabulary,
and `train_word2vec` trains the embeddings, on the shared implementation in
`libs/word2vec`.
"""

__all__: list[str] = []
