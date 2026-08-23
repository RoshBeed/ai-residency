"""CBOW word embeddings.

The towers in this repo do not learn word meaning from scratch; they sit on frozen
embeddings trained here first, on `text8`. `train_tokenizer` builds the vocabulary,
`train_word2vec` trains the embeddings, and `train_regressor` is a legacy demo carried
over from Week 1 that the retrieval pipeline does not use.
"""

__all__: list[str] = []
