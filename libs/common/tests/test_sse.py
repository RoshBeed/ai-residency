"""Tests for ai_residency_common.sse."""

import queue

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from ai_residency_common.sse import format_sse, stream_sse


def test_format_sse_encodes_a_dict_into_a_data_line() -> None:
    result = format_sse({"stage": "test", "value": 42})
    assert result == 'data: {"stage": "test", "value": 42}\n\n'


def test_format_sse_handles_string_values() -> None:
    result = format_sse({"stage": "digit", "digit": "7"})
    assert '"digit": "7"' in result
    assert result.startswith("data: ")
    assert result.endswith("\n\n")


def test_stream_sse_yields_the_events_posted_by_the_worker() -> None:
    def worker(events: queue.Queue[dict[str, str] | None]) -> None:
        events.put({"stage": "first"})
        events.put({"stage": "second"})

    app = FastAPI()

    @app.get("/stream")
    def stream() -> StreamingResponse:
        return stream_sse(worker)

    client = TestClient(app)
    response = client.get("/stream")
    assert response.status_code == 200

    parts = [p for p in response.text.strip().split("\n\n") if p]
    assert len(parts) == 2
    assert '"stage": "first"' in parts[0]
    assert '"stage": "second"' in parts[1]


def test_stream_sse_catches_exceptions_as_error_events() -> None:
    def worker(events: queue.Queue[dict[str, str] | None]) -> None:
        events.put({"stage": "before"})
        raise RuntimeError("boom")

    app = FastAPI()

    @app.get("/stream")
    def stream_second() -> StreamingResponse:
        return stream_sse(worker)

    client = TestClient(app)
    response = client.get("/stream")
    assert response.status_code == 200

    text = response.text
    assert '"stage": "before"' in text
    assert '"stage": "error"' in text
    assert "boom" in text
