"""What every service page needs: the framing around a demo, and the small helpers.

`app.py` used to hold this alongside a 892-line `if/elif` chain over the nine
services, all of it running at import. That is what kept the page flow untestable --
importing anything from it started a Streamlit script run. The chain is nine modules
in `views/` now, and this is the part they share.

Nothing here talks to a model. Pages are thin clients over each backend's already
verified endpoint contract: request/response wiring and display.
"""

import base64
import io
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

from client import NOT_READY_MESSAGE, Result, is_ready
from components import esc
from projects import API_URLS, PAGES, PROJECTS, tag_link

__all__ = [
    "bytes_to_b64",
    "canvas_to_png_bytes",
    "esc",
    "image_file_to_b64",
    "image_to_b64",
    "service_page",
    "show",
    "show_stream_error",
]


def show(result: Result) -> dict[str, Any] | None:
    """Return a successful call's body, or render why there isn't one.

    The three outcomes get three different treatments on purpose. A 503 is not an
    error the reader caused or can fix, and it is the expected answer from any service
    whose checkpoint has not been trained yet -- so it reads as information, not as a
    failure. Everything else is a real problem and says so.
    """
    if result.ok:
        return result.data
    if result.status == "not_ready":
        st.info(result.message)
    else:
        st.error(result.message)
    return None


def show_stream_error(error: requests.exceptions.RequestException, service: str) -> None:
    """Render a failed SSE stream the way `show` renders a failed call.

    `stream` calls `raise_for_status`, so a backend with no checkpoint arrives here as
    an HTTPError carrying a 503 -- the same "not trained yet" state the non-streaming
    path reports, and it should read the same way rather than as a raw exception.
    """
    response = getattr(error, "response", None)
    if response is not None and response.status_code == 503:
        st.info(NOT_READY_MESSAGE)
    else:
        st.error(f"Could not reach {service}: {error}")


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def image_to_b64(uploaded_file: UploadedFile) -> str:
    return bytes_to_b64(uploaded_file.getvalue())


def image_file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def canvas_to_png_bytes(image_data: np.ndarray) -> bytes:
    """Convert the drawable canvas's RGBA pixel array into grayscale PNG bytes."""
    image = Image.fromarray(image_data.astype("uint8"), "RGBA").convert("L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_brief(service: str) -> None:
    """Brief intro paragraph above the interactive demo."""
    project = PROJECTS[service]
    if "intro" in project:
        st.markdown(project["intro"])
    st.divider()


def _render_details(service: str) -> None:
    """Diagrams, stats, tags, and docs link below the demo."""
    project = PROJECTS[service]

    if project.get("diagrams"):
        diagrams = project["diagrams"]
        if len(diagrams) == 2:
            for col, (path, caption) in zip(st.columns(2), diagrams, strict=True):
                with col:
                    b64 = image_file_to_b64(path)
                    st.markdown(
                        '<div style="height:400px;display:flex;align-items:center;'
                        "justify-content:center;padding:16px;background:#fff;"
                        'border-radius:4px;">'
                        f'<img src="data:image/png;base64,{b64}" '
                        'style="max-width:100%;max-height:400px;object-fit:contain;">'
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(caption)
        else:
            for path, caption in diagrams:
                st.image(path, width=650)
                st.caption(caption)

    if project.get("model_specs"):
        st.markdown("**Model details**")
        for line in project["model_specs"]:
            st.markdown(f"- {line}")

    if project.get("stats"):
        cols = st.columns(len(project["stats"]))
        for col, (label, value) in zip(cols, project["stats"], strict=True):
            with col:
                st.metric(label, value)

    tags_html = "".join(tag_link(t) for t in project["techniques"])
    model_html = "".join(
        f'<a href="{url}" target="_blank" class="tag-model">{label}</a>'
        for label, url in project.get("model_links", [])
    )
    data_html = "".join(
        f'<a href="{url}" target="_blank" class="tag-data">{label}</a>'
        for label, url in project.get("datasets", [])
    )
    paper_html = "".join(
        f'<a href="{url}" target="_blank" class="tag-paper">{label}</a>'
        for label, _title, url in project.get("papers", [])
    )
    st.markdown(
        f'<div class="tag-row" style="margin:6px 0 12px;">'
        f"{tags_html}{model_html}{data_html}{paper_html}</div>",
        unsafe_allow_html=True,
    )

    docs_link = f"https://github.com/RoshBeed/ai-residency/blob/main/{project['docs_path']}"
    st.caption(
        f"[Full technical write-up →]({docs_link}) "
        "(architecture diagrams, method walkthrough, experimental results, references)"
    )


@contextmanager
def service_page(service: str) -> Iterator[None]:
    """Wrap one service's demo in the framing every service page shares.

    Title, readiness check and intro above; write-up, diagrams and stats below. Each
    view module is then only the part that differs -- the inputs it takes and how it
    renders that backend's response.

    An unready backend stops the script inside `__enter__`, so the body never runs and
    nothing below it renders. That is deliberate: a page whose backend is answering 503
    has nothing to show but the reason.
    """
    st.markdown(f"## {PAGES[service]}")

    if not is_ready(service):
        st.error(
            f"This project's backend isn't ready at `{API_URLS[service]}`. Either it isn't "
            f"running — start it with `docker compose up {service}` — or it started without "
            "a usable checkpoint, in which case its `/ready` endpoint is reporting 503 and "
            "every request would come back the same way."
        )
        st.stop()

    _render_brief(service)
    st.subheader("Try it out")

    yield

    # Below the demo, so the reader can try it before reading about it.
    _render_details(service)
