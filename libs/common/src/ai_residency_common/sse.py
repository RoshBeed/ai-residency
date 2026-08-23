"""Server-Sent Event helpers shared by every serving API in the portfolio.

Before this module existed, each of the six repos had its own copy of
``_sse()`` (two lines, character-for-character identical) and half of them
reimplemented the queue+thread+``StreamingResponse`` pattern for streaming
endpoints.
"""

import json
import queue
import threading
from collections.abc import Callable, Generator
from typing import Any

from fastapi.responses import StreamingResponse

__all__ = ["format_sse", "stream_sse"]


def format_sse(event: dict[str, Any]) -> str:
    """Format one Server-Sent Event line.

    Args:
        event: The event dict to serialise.

    Returns:
        A ``"data: {...}\\n\\n"`` string ready for a ``text/event-stream`` response.
    """
    return f"data: {json.dumps(event)}\n\n"


def stream_sse(
    worker: Callable[[queue.Queue[dict[str, Any] | None]], None],
) -> StreamingResponse:
    """Wrap a blocking worker function in a ``text/event-stream`` response.

    The worker runs on a daemon thread and receives a queue to post events onto.
    It must put ``None`` as a sentinel when finished. Exceptions in the worker
    are caught and emitted as ``{"stage": "error", "detail": ...}`` events.

    Args:
        worker: A callable that accepts one argument, a ``queue.Queue``. Post
            each event as a ``dict``, then ``None`` to signal completion.

    Returns:
        A FastAPI ``StreamingResponse`` with ``media_type="text/event-stream"``
        ready to return from an endpoint.
    """
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def _run() -> None:
        try:
            worker(events)
        except Exception as error:
            events.put({"stage": "error", "detail": str(error)})
        finally:
            events.put(None)

    threading.Thread(target=_run, daemon=True).start()

    def _generate() -> Generator[str, None, None]:
        while True:
            event = events.get()
            if event is None:
                break
            yield format_sse(event)

    return StreamingResponse(_generate(), media_type="text/event-stream")
