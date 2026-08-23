"""Interactive retrieval CLI.

All the work is in `query_engine.run_query`; this is the prompt loop around it. It used
to patch the import path by hand to make its own package importable, which is what running
a file by path instead of as a module costs you. It is
`python -m retrieval.query_redis_ann` now.
"""

from ai_residency_common import get_logger

from config import Config
from retrieval.query_engine import run_query

__all__ = ["main"]

logger = get_logger(__name__)


def main() -> None:
    """Read queries from stdin until EOF or `exit`, printing the top results."""
    config = Config()
    while True:
        try:
            query = input("Enter your query (or 'exit' to quit): ").strip()
        except EOFError:
            break
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = run_query(query, top_k=config.top_k)
        if not results:
            logger.info("No results found.")
            continue

        logger.info("Top %d results:", len(results))
        for rank, (score, text, _emb) in enumerate(results, start=1):
            logger.info("Rank %d  distance=%.4f\n%s\n---", rank, score, text)


if __name__ == "__main__":
    main()
