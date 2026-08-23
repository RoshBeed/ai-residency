"""Indexing and querying.

`save_doc_embeddings_to_redis` embeds the corpus once with the trained document tower
and writes it into RediSearch. `query_engine` embeds a query with the trained query
tower and searches that index. `query_redis_ann` is the interactive CLI on top.
"""

__all__: list[str] = []
