"""MS MARCO to tokenized training triples.

`generate_triples` builds (query, positive, negative) triples from MS MARCO and
`tokenize_triples` maps them through the CBOW vocabulary. The rest are exploratory or
transfer scripts the pipeline does not depend on.
"""

__all__: list[str] = []
