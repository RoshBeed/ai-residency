"""One way to call a backend, and one way to describe what came back.

Every page used to inline the same six lines -- post, `raise_for_status`, catch
`RequestException`, `st.error(str(exception))` -- eleven times over, and they did not
agree with each other. Most surfaced a raw exception string to the reader. Only one
treated 503 as a distinct state rather than a failure, which is backwards: *every*
service in this portfolio answers 503 when it has no trained checkpoint, so that is the
single most likely non-success response and the one worth explaining well.

`call` collapses that into three outcomes a page can branch on:

* `ok` -- the body is in `data`
* `not_ready` -- the backend is up but has no usable checkpoint (503)
* `failed` -- unreachable, timed out, or returned some other error

Nothing here imports Streamlit, so it can be tested without a Streamlit runtime.
"""

import json
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Literal

import requests

from projects import API_URLS

__all__ = ["Result", "call", "is_ready", "stream"]

Status = Literal["ok", "not_ready", "failed"]

# What a backend that has no checkpoint says, in the frontend's own words. The API's
# own 503 detail is written for a machine; this is the sentence a reader needs.
NOT_READY_MESSAGE = (
    "This service is running but has no trained checkpoint loaded, so it cannot answer "
    "yet. The pipeline, trainer and API are wired up end to end -- what is missing is a "
    "trained model, not the plumbing."
)


@dataclass(frozen=True)
class Result:
    """The outcome of one backend call."""

    status: Status
    data: dict[str, Any] | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        """Whether `data` is populated."""
        return self.status == "ok"


def call(
    service: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: int = 15,
) -> Result:
    """POST to one backend and classify the response.

    Args:
        service: Key into `API_URLS`, e.g. `"word-embeddings"`.
        path: Endpoint path beginning with a slash, e.g. `"/nearest"`.
        payload: JSON body. None sends an empty object.
        timeout: Seconds to wait. Raise it for endpoints that run a model per
            request rather than serving a cached one.

    Returns:
        A `Result` whose `status` is `ok`, `not_ready` or `failed`.
    """
    url = f"{API_URLS[service]}{path}"
    try:
        response = requests.post(url, json=payload or {}, timeout=timeout)
    except requests.exceptions.Timeout:
        return Result("failed", message=f"{service} did not respond within {timeout}s.")
    except requests.exceptions.RequestException as error:
        return Result("failed", message=f"Could not reach {service} at {url}: {error}")

    if response.status_code == 503:
        return Result("not_ready", message=NOT_READY_MESSAGE)

    if not response.ok:
        return Result(
            "failed",
            message=f"{service} returned {response.status_code}: {_detail(response)}",
        )

    try:
        return Result("ok", data=response.json())
    except ValueError:
        return Result("failed", message=f"{service} returned a body that is not JSON.")


def _detail(response: requests.Response) -> str:
    """Pull FastAPI's `detail` out of an error body, falling back to the raw text."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return response.text[:200]


def stream(
    service: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: int = 120,
) -> Generator[dict[str, Any], None, None]:
    """POST to an SSE endpoint and yield each event as a parsed dict.

    A thin wrapper over `requests`' own streaming support -- no extra dependency.
    Each backend event is one `data: {...}` line followed by a blank line; this reads
    line by line and parses only the `data:` ones, which is sufficient for the
    single-line JSON payloads these backends emit.
    """
    url = f"{API_URLS[service]}{path}"
    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        # Decode explicitly rather than via decode_unicode=True: the flag does yield
        # str at run time, but requests' stubs type iter_lines as bytes either way, so
        # the str comparison below would have no basis under strict mypy.
        for raw in response.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if line.startswith("data: "):
                yield json.loads(line[len("data: ") :])


def is_ready(service: str) -> bool:
    """Whether a backend can actually answer, not merely whether its process is up.

    `/ready` rather than `/health`: health answers 200 for a service that started but
    has no usable checkpoint, so an indicator built on it would show green for a
    backend that 503s every request.
    """
    try:
        return requests.get(f"{API_URLS[service]}/ready", timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False
