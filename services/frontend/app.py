"""
ML systems portfolio -- a single Streamlit frontend that calls every
backend APIs directly over HTTP. Each page below is a thin client around one
project's real, already-verified endpoint contract -- no inference logic lives
here, only request/response wiring and display.

Every page also renders a real trace of what the backend actually did to
produce its answer (which words were recognised, per-token probabilities,
a live fine-tuning loss curve, the PPO policy's sampling candidates at each
generation step, ...) rather than only the final result. That data comes
straight from each project's own API response -- nothing here recomputes or
simulates it.
"""

import base64
import io
import math
from itertools import pairwise
from typing import Any

import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_drawable_canvas import st_canvas

from client import NOT_READY_MESSAGE, Result, call, is_ready, stream
from components import (
    draw_patch_grid,
    esc,
    render_diff,
    render_digit_trace,
    render_diverging_token_heatmap,
    render_loss_curve,
    render_probability_bars,
    render_stats,
    render_token_heatmap,
    render_word_trace,
    seq_step,
    text_color_for,
)
from projects import API_URLS, PAGES, PROJECTS, tag_link
from theme import LOGO_URI, PALETTE_CSS

st.set_page_config(page_title="ML Systems Portfolio", page_icon="🧩", layout="wide")

st.logo(LOGO_URI)


st.markdown(PALETTE_CSS, unsafe_allow_html=True)


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


def show_stream_error(error: requests.exceptions.RequestException) -> None:
    """Render a failed SSE stream the way `show` renders a failed call.

    `stream` calls `raise_for_status`, so a backend with no checkpoint arrives here as
    an HTTPError carrying a 503 -- the same "not trained yet" state the non-streaming
    path reports, and it should read the same way rather than as a raw exception.
    """
    response = getattr(error, "response", None)
    if response is not None and response.status_code == 503:
        st.info(NOT_READY_MESSAGE)
    else:
        st.error(f"Could not reach {_PAGE}: {error}")


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


def render_service_brief(service: str) -> None:
    """Brief intro paragraph above the interactive demo."""
    project = PROJECTS[service]
    if "intro" in project:
        st.markdown(project["intro"])
    st.divider()


def render_service_details(service: str) -> None:
    """Diagrams, stats, tags, and docs link below the demo."""
    project = PROJECTS[service]

    if project.get("diagrams"):
        diagrams = project["diagrams"]
        if len(diagrams) == 2:
            c1, c2 = st.columns(2)
            with c1:
                b64 = image_file_to_b64(diagrams[0][0])
                st.markdown(
                    f'<div style="height:400px;display:flex;align-items:center;justify-content:center;padding:16px;background:#fff;border-radius:4px;">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:100%;max-height:400px;object-fit:contain;">'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(diagrams[0][1])
            with c2:
                b64 = image_file_to_b64(diagrams[1][0])
                st.markdown(
                    f'<div style="height:400px;display:flex;align-items:center;justify-content:center;padding:16px;background:#fff;border-radius:4px;">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:100%;max-height:400px;object-fit:contain;">'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(diagrams[1][1])
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
        f'<div class="tag-row" style="margin:6px 0 12px;">{tags_html}{model_html}{data_html}{paper_html}</div>',
        unsafe_allow_html=True,
    )

    docs_link = f"https://github.com/RoshBeed/ai-residency/blob/main/{project['docs_path']}"
    st.caption(
        f"[Full technical write-up →]({docs_link}) "
        "(architecture diagrams, method walkthrough, experimental results, references)"
    )


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------


def render_home_page() -> None:
    st.markdown(
        "<h2 style='text-align:center;margin:0 0 2px;font-size:1.2rem;'>ML Systems Portfolio</h2>"
        "<hr style='margin:0 0 6px;border:none;border-top:1px solid var(--border);'>",
        unsafe_allow_html=True,
    )

    # Project cards — the main focus, above the fold.
    service_keys = list(PROJECTS.keys())
    for row_start in range(0, len(service_keys), 3):
        row = service_keys[row_start : row_start + 3]
        cols = st.columns(len(row), gap="medium")
        for i, wk in enumerate(row):
            p = PROJECTS[wk]
            with cols[i]:
                tags_html = "".join(tag_link(t) for t in p["techniques"])
                data_html = "".join(
                    f'<a href="{url}" target="_blank" class="tag-data">{label}</a>'
                    for label, url in p.get("datasets", [])
                )
                model_html = "".join(
                    f'<a href="{url}" target="_blank" class="tag-model">{label}</a>'
                    for label, url in p.get("model_links", [])
                )
                paper_html = "".join(
                    f'<a href="{url}" target="_blank" class="tag-paper" title="{label}: {title}">{title[:16] + "…" if len(title) > 16 else title}</a>'
                    for label, title, url in p.get("papers", [])
                )
                st.markdown(
                    f'<div class="card">'
                    f'<div class="card-header">'
                    f'<span class="card-icon">{p["icon"]}</span>'
                    f'<span class="card-title">{p["title"]}</span>'
                    f"</div>"
                    f'<div class="card-desc">{p["description"]}</div>'
                    f'<div class="tag-row">{tags_html}{model_html}{data_html}{paper_html}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button("🔬 Open", key=f"try_{wk}", use_container_width=True):
                    navigate(wk)
                    st.rerun()

    st.markdown(
        "<p style='text-align:center;font-size:0.85rem;color:var(--text-muted);margin-top:8px;'>"
        "End-to-end ML pipelines spanning NLP, computer vision, speech, and reinforcement learning."
        "</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state + routing setup
# ---------------------------------------------------------------------------

if "page" in st.query_params and st.query_params["page"] != st.session_state.get("page"):
    st.session_state.page = st.query_params["page"]
if "page" not in st.session_state:
    st.session_state.page = "home"


def navigate(page: str) -> None:
    st.session_state.page = page
    st.query_params["page"] = page


_PAGE = st.session_state.page


# ---------------------------------------------------------------------------
# Sidebar: navigation + tech-depth panel + backend status
# ---------------------------------------------------------------------------

with st.sidebar:
    if st.button("← Home", key="sidebar_home", type="tertiary"):
        navigate("home")
        st.rerun()

    if _PAGE != "home":
        st.divider()
        for wk, label in PAGES.items():
            if wk == _PAGE:
                st.markdown(f"{PROJECTS[wk]['icon']} **{label}**")
            else:
                if st.button(f"{PROJECTS[wk]['icon']} {label}", key=f"nav_{wk}", type="tertiary"):
                    navigate(wk)
                    st.rerun()

    st.divider()
    st.caption("[GitHub →](https://github.com/RoshBeed/ai-residency/tree/main/services/frontend)")
    st.caption("[LinkedIn →](https://linkedin.com/in/RoshBeed)")


# ---------------------------------------------------------------------------
# Main content routing
# ---------------------------------------------------------------------------

if _PAGE == "home":
    render_home_page()
    st.stop()

# --- Every service page shares the title + readiness check ---

st.markdown(f"## {PAGES[_PAGE]}")

if not is_ready(_PAGE):
    st.error(
        f"This project's backend isn't ready at `{API_URLS[_PAGE]}`. Either it isn't "
        f"running — start it with `docker compose up {_PAGE}` — or it started without "
        "a usable checkpoint, in which case its `/ready` endpoint is reporting 503 and "
        "every request would come back the same way."
    )
    st.stop()

render_service_brief(_PAGE)
st.subheader("Try it out")

# ---------------------------------------------------------------------------
# Word Embeddings (nearest-neighbour + analogy)
# ---------------------------------------------------------------------------
if _PAGE == "word-embeddings":
    st.markdown("**Nearest neighbours** — the words closest to a query word in the learned space.")
    word = st.text_input("Word", value="python")
    if st.button("Find nearest", type="primary") and word:
        data = show(call(_PAGE, "/nearest", {"word": word}))
        if data:
            st.markdown(
                '<div class="trace-caption">'
                f'Loaded from checkpoint source "{data["checkpoint_source"]}".</div>',
                unsafe_allow_html=True,
            )
            for rank, (neighbour, similarity) in enumerate(data["nearest"], start=1):
                st.markdown(
                    f'**{rank}.** `{neighbour}` <span class="dim">(cosine {similarity:.3f})</span>',
                    unsafe_allow_html=True,
                )

    st.divider()
    st.markdown("**Analogy** — find the word closest to `a − b + c`.")
    c1, c2, c3 = st.columns(3)
    with c1:
        a = st.text_input("a", value="king")
    with c2:
        b = st.text_input("b", value="man")
    with c3:
        c = st.text_input("c", value="woman")
    if st.button("Solve analogy", type="primary") and a and b and c:
        data = show(call(_PAGE, "/analogy", {"a": a, "b": b, "c": c}))
        if data:
            for neighbour, similarity in data["nearest"]:
                st.markdown(
                    f"`{a} − {b} + {c}` ≈ **{neighbour}** "
                    f'<span class="dim">(cosine {similarity:.3f})</span>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Content Scoring (regression head over frozen embeddings)
# ---------------------------------------------------------------------------
elif _PAGE == "content-scoring":
    title = st.text_input(
        "Hacker News post title",
        value="Show HN: I built a CLI tool that writes your commit messages",
    )
    if st.button("Predict score", type="primary") and title:
        st.caption("Live -- each stage below arrives the moment it completes:")
        tracker = st.empty()
        live_words = st.empty()
        done_lines: list[str] = []
        try:
            for event in stream(
                _PAGE,
                "/predict/stream",
                {"title": title},
                timeout=15,
            ):
                stage = event["stage"]
                if stage == "word_lookup":
                    word_trace = event["word_trace"]
                    known_c = sum(1 for w in word_trace if w["known"])
                    total_c = len(word_trace)
                    done_lines.append(
                        f'<span class="done">✓</span> Word lookup: {known_c}/{total_c} recognised'
                    )
                    done_lines.append('<span class="active">⟳</span> Building embedding vector…')
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(done_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    live_words.markdown(render_word_trace(word_trace), unsafe_allow_html=True)
                elif stage == "embedding":
                    norm = event["title_embedding_norm"]
                    done_lines[-1] = (
                        f'<span class="done">✓</span> Embedding built (norm = {norm:.3f})'
                    )
                    done_lines.append('<span class="active">⟳</span> Predicting score…')
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(done_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                elif stage == "done":
                    data = event
                    done_lines[-1] = (
                        f'<span class="done">✓</span> Predicted {data["predicted_score"]:.1f} upvotes'
                    )
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(done_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="trace-caption">'
                        "Words looked up in CBOW vocabulary (258,196 words). "
                        "Unknown words are skipped.</div>",
                        unsafe_allow_html=True,
                    )
                    live_words.empty()
                    st.markdown(render_word_trace(data["word_trace"]), unsafe_allow_html=True)
                    known = sum(1 for w in data["word_trace"] if w["known"])
                    st.markdown(
                        render_stats(
                            [
                                ("Words recognised", f"{known} / {len(data['word_trace'])}"),
                                ("Embedding dim", str(data["embedding_dim"])),
                                ("Title vector norm", f"{data['title_embedding_norm']:.3f}"),
                            ]
                        ),
                        unsafe_allow_html=True,
                    )
                    st.metric("Predicted score", f"{data['predicted_score']:.1f} upvotes")
                    with st.expander("Raw response"):
                        st.json(data)
                elif stage == "error":
                    tracker.empty()
                    st.error(f"Prediction failed: {event['detail']}")
        except requests.exceptions.RequestException as error:
            show_stream_error(error)

# ---------------------------------------------------------------------------
# RAG & Semantic Search
# ---------------------------------------------------------------------------
elif _PAGE == "vector-search-retrieval":
    query = st.text_input("Search query", value="was ronald reagan a democrat")
    top_k = st.slider("Top K", 1, 10, 5)
    if st.button("Search", type="primary") and query:
        st.caption("Live -- each stage below arrives the moment it completes:")
        tracker = st.empty()
        live_words = st.empty()
        retrieved_lines: list[str] = []
        try:
            for event in stream(
                _PAGE,
                "/query/stream",
                {"text": query, "top_k": top_k},
                timeout=15,
            ):
                stage = event["stage"]
                if stage == "vocabulary":
                    wt = event["word_trace"]
                    known_c = sum(1 for w in wt if w["known"])
                    retrieved_lines.append(
                        f'<span class="done">✓</span> Vocabulary check: '
                        f"{known_c}/{len(wt)} words recognised"
                    )
                    retrieved_lines.append(
                        '<span class="active">⟳</span> Encoding query with GRU tower…'
                    )
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(retrieved_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    live_words.markdown(render_word_trace(wt), unsafe_allow_html=True)
                elif stage == "encoded":
                    norm = event["embedding_norm"]
                    retrieved_lines[-1] = (
                        f'<span class="done">✓</span> Query encoded (norm = {norm:.3f})'
                    )
                    retrieved_lines.append(
                        '<span class="active">⟳</span> Searching Redis via HNSW…'
                    )
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(retrieved_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                elif stage == "done":
                    data = event
                    n_res = len(data["results"])
                    retrieved_lines[-1] = f'<span class="done">✓</span> {n_res} results retrieved'
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(retrieved_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="trace-caption">'
                        "Words encoded by trained GRU query tower, then compared against "
                        "indexed documents via cosine distance (lower = closer).</div>",
                        unsafe_allow_html=True,
                    )
                    live_words.empty()
                    st.markdown(render_word_trace(data["word_trace"]), unsafe_allow_html=True)
                    st.markdown(
                        render_stats([("Query vector norm", f"{data['embedding_norm']:.3f}")]),
                        unsafe_allow_html=True,
                    )
                    if not data["results"]:
                        st.info("No results — the index may be empty on this deployment.")
                    else:
                        chart_data = {f"#{r['rank']}": r["score"] for r in data["results"]}
                        st.bar_chart(
                            chart_data,
                            horizontal=True,
                            x_label="cosine distance",
                            y_label="result rank",
                        )
                    for res in data["results"]:
                        st.markdown(
                            f'<div class="score-detail"><strong>#{res["rank"]}</strong> '
                            f"(cos-dist {res['score']:.4f})</div>"
                            f'<div class="score-detail">{esc(res["text"])}</div>',
                            unsafe_allow_html=True,
                        )
                    with st.expander("Raw response"):
                        st.json(data)
                elif stage == "error":
                    tracker.empty()
                    st.error(f"Retrieval failed: {event['detail']}")
        except requests.exceptions.RequestException as error:
            show_stream_error(error)

# ---------------------------------------------------------------------------
# Vision Transformer & Seq2Seq
# ---------------------------------------------------------------------------
elif _PAGE == "vit-classifier":
    col_draw, col_upload = st.columns(2)
    with col_draw:
        st.caption("Draw a digit")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=18,
            stroke_color="#ffffff",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="vit_classifier_canvas",
        )
    with col_upload:
        st.caption("Or upload one")
        uploaded = st.file_uploader(
            "Upload a digit image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.image(uploaded, width=150)

    image_bytes: bytes | None = None
    if canvas_result.image_data is not None:
        image_bytes = canvas_to_png_bytes(canvas_result.image_data)
    elif uploaded is not None:
        image_bytes = uploaded.getvalue()

    if image_bytes is not None and st.button("Classify", type="primary"):
        b64_image = bytes_to_b64(image_bytes)
        vit_data = show(call(_PAGE, "/predict", {"image_data": b64_image}))

        if vit_data:
            st.caption(
                f"Input: {vit_data['img_size']}x{vit_data['img_size']}px, "
                f"{vit_data['patch_size']}x{vit_data['patch_size']} patches."
            )
            st.image(
                draw_patch_grid(image_bytes, vit_data["img_size"], vit_data["patch_size"]),
                width=260,
            )
            st.markdown(f"**Predicted: {vit_data['pred']}**")
            if vit_data.get("class_probabilities"):
                st.markdown(
                    render_probability_bars(vit_data["class_probabilities"], vit_data["pred"]),
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# ViT Seq2Seq -- multi-digit, decoded one token at a time
# ---------------------------------------------------------------------------
elif _PAGE == "vit-seq2seq":
    col_draw, col_upload = st.columns(2)
    with col_draw:
        st.caption("Draw a multi-digit number")
        seq_canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=14,
            stroke_color="#ffffff",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="vit_seq2seq_canvas",
        )
    with col_upload:
        st.caption("Or upload one")
        seq_uploaded = st.file_uploader(
            "Upload a multi-digit image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if seq_uploaded is not None:
            st.image(seq_uploaded, width=150)

    seq_bytes: bytes | None = None
    if seq_canvas.image_data is not None:
        seq_bytes = canvas_to_png_bytes(seq_canvas.image_data)
    elif seq_uploaded is not None:
        seq_bytes = seq_uploaded.getvalue()

    if seq_bytes is not None and st.button("Read the number", type="primary"):
        b64_seq = bytes_to_b64(seq_bytes)
        st.caption("Input: 84x84px, 16x16 patches. Decoded autoregressively.")
        st.image(draw_patch_grid(seq_bytes, 84, 16), width=260)

        st.caption("Live -- each digit below appears the moment the model emits it:")
        digit_display = st.empty()
        seq2seq_data = None
        digit_trace_sofar: list[dict[str, Any]] = []
        try:
            for event in stream(
                _PAGE,
                "/predict/stream",
                {"image_data": b64_seq},
                timeout=30,
            ):
                stage = event["stage"]
                if stage == "digit" and event["digit"]:
                    digit_trace_sofar.append(
                        {"digit": event["digit"], "confidence": event["confidence"]}
                    )
                    digit_display.markdown(
                        render_digit_trace(digit_trace_sofar),
                        unsafe_allow_html=True,
                    )
                elif stage == "done":
                    seq2seq_data = event
                    digit_display.empty()
                elif stage == "error":
                    digit_display.empty()
                    st.error(f"Seq2seq generation failed: {event['detail']}")
        except requests.exceptions.RequestException as e:
            st.error(f"SSE request failed: {e}")

        if seq2seq_data:
            st.markdown(f"**Predicted: {seq2seq_data['pred']}**")
            if seq2seq_data.get("digit_trace"):
                st.caption("Darker = higher confidence at each step:")
                st.markdown(render_digit_trace(seq2seq_data["digit_trace"]), unsafe_allow_html=True)
            with st.expander("Raw SSE response"):
                st.json(seq2seq_data)

# ---------------------------------------------------------------------------
# Multimodal Transfer Learning
# ---------------------------------------------------------------------------
elif _PAGE == "multimodal-captioning":
    col_upload, col_camera = st.columns(2)
    with col_upload:
        uploaded = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
        )
        if uploaded is not None:
            st.image(uploaded, width=250)
    with col_camera:
        camera = st.camera_input("Take a photo", label_visibility="collapsed")
        if camera is not None:
            st.image(camera, width=250)

    image_source = uploaded or camera
    if image_source is not None and st.button("Generate caption", type="primary"):
        st.caption("Live -- each token below appears the moment the model emits it:")
        tracker = st.empty()
        live_heatmap = st.empty()
        tokens: list[dict[str, Any]] = []
        data = None
        try:
            for event in stream(
                _PAGE,
                "/caption/stream",
                {"image_data": image_to_b64(image_source), "checkpoint_name": "adapter.pth"},
                timeout=90,
            ):
                stage = event["stage"]
                if stage == "model_loaded":
                    tracker.markdown(
                        '<div class="tracker">'
                        '<span class="done">✓</span> CLIP + Qwen3 loaded<br>'
                        '<span class="active">⟳</span> Generating caption…'
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif stage == "token":
                    tokens.append({"token": event["token"], "probability": event["probability"]})
                    live_heatmap.markdown(render_token_heatmap(tokens), unsafe_allow_html=True)
                    tracker.markdown(
                        f'<div class="tracker">'
                        f'<span class="done">✓</span> Models loaded<br>'
                        f'<span class="active">⟳</span> Token {event["step"] + 1}: '
                        f'"<strong>{esc(event["token"])}</strong>" '
                        f"({event['probability'] * 100:.1f}% confidence)"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif stage == "done":
                    data = event
                    tracker.markdown(
                        f'<div class="tracker">'
                        f'<span class="done">✓</span> Models loaded<br>'
                        f'<span class="done">✓</span> {len(data["token_trace"])} tokens generated'
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif stage == "error":
                    tracker.empty()
                    st.error(f"Captioning failed: {event['detail']}")
        except requests.exceptions.RequestException as error:
            show_stream_error(error)

        if data:
            st.caption(
                "CLIP image embedding → prepended to prompt → Qwen3 autoregressive decode. "
                "Darker = higher sampling probability."
            )
            live_heatmap.empty()
            st.markdown(render_token_heatmap(data["token_trace"]), unsafe_allow_html=True)
            st.success(data["caption"])
            st.caption(f"Checkpoint: {data['checkpoint_used']}")
            with st.expander("Raw SSE response"):
                st.json(data)

# ---------------------------------------------------------------------------
# Emotion-Aware Speech Recognition
# ---------------------------------------------------------------------------
elif _PAGE == "whisper-fine-tuning":
    col_record, col_upload = st.columns(2)
    with col_record:
        audio = st.audio_input("Record audio")
        if audio is not None:
            st.audio(audio)
    with col_upload:
        audio_file = st.file_uploader(
            "Upload audio", type=["mp3", "wav", "m4a", "aiff", "flac"], label_visibility="collapsed"
        )
        if audio_file is not None:
            st.audio(audio_file)

    audio_source = audio or audio_file
    audio_bytes: bytes | None = audio_source.getvalue() if audio_source else None

    target_text = st.text_input(
        "Fine-tune target text",
        value="My name is Rosh",
        help="What the model is fine-tuned to transcribe the clip as. The default is "
        "the demo's own ground truth for the bundled 'My name is Rosh' clip -- if you "
        "record something else, try changing this to match it, or leave it as-is to "
        "see what happens when the target doesn't match what was actually said.",
    )
    if audio_bytes is not None and st.button("Transcribe", type="primary"):
        st.caption("Live -- each event below arrives the moment that stage finishes:")
        tracker = st.empty()
        live_curve = st.empty()
        live_table = st.empty()
        baseline_loss = None
        base_output = None
        transcript_lines: list[str] = []
        steps: list[dict[str, Any]] = []
        try:
            for event in stream(
                _PAGE,
                "/transcribe/stream",
                {"audio_data": bytes_to_b64(audio_bytes), "target_text": target_text},
                timeout=120,
            ):
                stage = event["stage"]
                if stage == "base_transcript":
                    base_output = event["text"]
                    transcript_lines.append(
                        f'<span class="done">✓</span> Base whisper: "<em>{esc(base_output)}</em>"'
                    )
                    transcript_lines.append(
                        '<span class="active">⟳</span> Computing baseline loss…'
                    )
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(transcript_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                elif stage == "baseline_loss":
                    baseline_loss = event["loss"]
                    transcript_lines[-1] = (
                        f'<span class="done">✓</span> Baseline loss: {baseline_loss:.4f}'
                    )
                    transcript_lines.append('<span class="active">⟳</span> Step 1/5…')
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(transcript_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                elif stage == "step":
                    steps.append(event)
                    transcript_lines[-1] = (
                        f'<span class="done">✓</span> Step {event["step"]}/5: '
                        f"loss {event['loss']:.4f}"
                    )
                    if event["step"] < 5:
                        transcript_lines.append(
                            f'<span class="active">⟳</span> Step {event["step"] + 1}/5…'
                        )
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(transcript_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    points = [(0, baseline_loss)] + [(s["step"], s["loss"]) for s in steps]
                    live_curve.markdown(render_loss_curve(points), unsafe_allow_html=True)
                    live_table.dataframe(
                        [{"step": 0, "loss": baseline_loss, "text": base_output}, *steps],
                        use_container_width=True,
                        hide_index=True,
                    )
                elif stage == "done":
                    st.session_state["audio-fine-tuning_data"] = event
                    tracker.empty()
                    live_curve.empty()
                    live_table.empty()
                elif stage == "error":
                    tracker.empty()
                    st.error(f"Fine-tuning failed: {event['detail']}")
        except requests.exceptions.RequestException as error:
            show_stream_error(error)

    data = st.session_state.get("audio-fine-tuning_data")
    if data:
        st.markdown(
            '<div class="trace-caption">'
            "Not a cached demo. Fine-tunes Whisper-tiny from scratch each request — "
            f'5 gradient steps toward target "{esc(target_text)}".</div>',
            unsafe_allow_html=True,
        )
        points = [(0, data["baseline_loss"])] + [
            (s["step"], s["loss"]) for s in data["fine_tune_trace"]
        ]
        st.markdown(render_loss_curve(points), unsafe_allow_html=True)
        st.markdown(
            render_stats(
                [
                    ("Baseline loss (step 0)", f"{data['baseline_loss']:.4f}"),
                    (
                        "Final loss",
                        f"{data['fine_tune_trace'][-1]['loss']:.4f}"
                        if data["fine_tune_trace"]
                        else "n/a",
                    ),
                ]
            ),
            unsafe_allow_html=True,
        )

        step_rows = [{"step": 0, "loss": data["baseline_loss"], "text": data["base_output"]}] + [
            {"step": s["step"], "loss": s["loss"], "text": s["text"]}
            for s in data["fine_tune_trace"]
        ]
        st.caption("Per-step transcript — green = added, red = dropped:")
        for prev_row, curr_row in pairwise(step_rows):
            loss = curr_row["loss"]
            st.markdown(f"**Step {curr_row['step']}** (loss = {loss:.4f})")
            st.markdown(render_diff(prev_row["text"], curr_row["text"]), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Base Whisper**")
            st.text(data["base_output"])
        with col2:
            st.markdown("**Fine-tuned (5 steps)**")
            st.text(data["ft_output"])
        with st.expander("Raw response"):
            st.json(data)


# ---------------------------------------------------------------------------
# Speech Emotion Recognition -- the emotion is a token the decoder emits
# ---------------------------------------------------------------------------
elif _PAGE == "speech-emotion-recognition":
    st.caption(
        "Upload a clip. The model predicts which of six emotions it hears, by scoring the "
        "emotion tokens its decoder can emit."
    )
    emotion_upload = st.file_uploader(
        "Upload audio (WAV/MP3)", type=["wav", "mp3", "m4a"], label_visibility="collapsed"
    )
    if emotion_upload is not None:
        st.audio(emotion_upload)

    if emotion_upload is not None and st.button("Predict emotion", type="primary"):
        # 120s: this one loads Whisper and runs it per request.
        result = show(
            call(
                _PAGE,
                "/predict",
                {"audio_data": bytes_to_b64(emotion_upload.getvalue())},
                timeout=120,
            )
        )
        if result:
            st.markdown(f"**Predicted: {result['emotion']}**")
            st.caption(f"Checkpoint source: `{result['checkpoint_source']}`")
            st.markdown(
                render_probability_bars(result["scores"], result["emotion"]),
                unsafe_allow_html=True,
            )
            with st.expander("Raw response"):
                st.json(result)

# ---------------------------------------------------------------------------
# RLHF for Text Summarization
# ---------------------------------------------------------------------------
elif _PAGE == "rlhf-ppo":
    ppo_steps = st.number_input("PPO steps", min_value=1, max_value=20, value=5, step=1)
    _PPO_STAGE_LABELS = {
        "generate": "1. Sampling",
        "kl_divergence": "2. Token-Level KL Penalty",
        "rewards": "3. Reward Scoring",
        "adjusted_reward": "4. KL-Penalized Reward",
        "value": "5. Value Function",
        "td_error": "6. Temporal Difference Error",
        "advantage": "7. Generalized Advantage Estimation",
        "reward_to_go": "8. Rewards-to-Go",
        "value_loss": "9. Value Function Loss",
        "clip": "10. Clip",
        "policy_loss": "11. PPO-Clip Loss",
        "policy_ratio": "12. Policy Ratio",
    }
    _PPO_STAGE_FIELD = {
        "kl_divergence": "kl_div",
        "adjusted_reward": "adjusted_reward",
        "value": "value",
        "td_error": "td_error",
        "advantage": "advantage",
        "reward_to_go": "reward_to_go",
        "clip": "clipped_ratio",
        "policy_ratio": "policy_ratio",
    }
    if st.button("Run PPO steps", type="primary"):
        st.caption("Live — 12 stages per step. Token rows persist across stages.")
        _rendering = {
            "kl_divergence": ("2. Token-Level KL Penalty", "kl_div"),
            "adjusted_reward": ("4. KL-Penalized Reward", "adjusted_reward"),
            "value": ("5. Value Function", "value"),
            "td_error": ("6. Temporal Difference Error", "td_error"),
            "advantage": ("7. Generalized Advantage Estimation", "advantage"),
            "reward_to_go": ("8. Rewards-to-Go", "reward_to_go"),
            "clip": ("10. Clip", "clipped_ratio"),
            "policy_ratio": ("12. Policy Ratio", "policy_ratio"),
        }

        _MATH = {
            "generate": "a_t  ∼  π_θ(·∣s_t)",
            "kl_divergence": "log π_θ(a_t∣s_t) / π_ref(a_t∣s_t)",
            "rewards": "r_ψ(x, y)",
            "adjusted_reward": "r − η·KL_t",
            "value": "V_ϕ(s_t)",
            "td_error": "δ_t  =  r_t + γV_ϕ(s_{t+1}) − V_ϕ(s_t)",
            "advantage": "Â_t  =  Σ_{l≥0} (γλ)^l δ_{t+l}",
            "reward_to_go": "R̂_t  =  Â_t + V_ϕ(s_t)",
            "value_loss": "L_V  =  E[(V_ϕ − R̂_t)²]    ···  ∂V",
            "clip": "clamp(r_t(θ), 1±ε)",
            "policy_loss": "L_CLIP  =  E[min(r_tÂ_t, clip(r_t, 1±ε)Â_t)]    ···  ∂π",
            "policy_ratio": "r_t(θ)  =  π_θ / π_old    ——  readout",
        }

        try:
            status_line = st.empty()
            unified = st.empty()
            blocks: list[str] = []
            last_load: str = ""
            sampling_chips: list[str] = []

            # Per-step metrics for the post-run summary card.
            step_records: list[dict[str, float | str]] = []
            cur: dict[str, float | str] = {}

            def _render_heatmap(
                label: str, stage: str, field: str, event: dict[str, Any], math: dict[str, Any]
            ) -> str:
                html = (
                    f'<div class="stage-block">'
                    f'<span class="stage-label">{label}</span>'
                    f'<span class="stage-math">{math[stage]}</span>'
                    f"</div>"
                )
                pairs_by_prompt: dict[int, list[dict[str, Any]]] = {}
                for pair in event["sa_pairs"]:
                    pairs_by_prompt.setdefault(pair["prompt_idx"], []).append(pair)
                for pidx in sorted(pairs_by_prompt):
                    ordered = sorted(pairs_by_prompt[pidx], key=lambda p: p["step"])
                    html += render_diverging_token_heatmap(
                        [p["action_token"] for p in ordered],
                        [p[field] for p in ordered],
                    )
                return html

            def flush() -> None:
                unified.markdown("".join(blocks), unsafe_allow_html=True)

            for event in stream(
                _PAGE,
                "/ppo_step/stream",
                {
                    "prompts": [
                        "SUBREDDIT: r/relationships\nTITLE: Should I admit to snooping?\nPOST: ...\nTL;DR:"
                    ],
                    "ppo_steps": ppo_steps,
                },
                timeout=600,
            ):
                stage = event["stage"]
                if stage == "loading":
                    last_load = event["detail"]
                    status_line.markdown(f"⏳ {last_load}")
                elif stage == "step_boundary":
                    status_line.empty()
                    if cur:
                        step_records.append(cur)
                        cur = {}
                    step_num = event["step"]
                    step_total = event["total"]
                    blocks.append(
                        f'<div class="step-header">PPO Step {step_num} / {step_total}</div>'
                    )
                    flush()
                elif stage == "generating":
                    prob = math.exp(event["logprob"])
                    bg = seq_step(prob)
                    fg = text_color_for(bg)
                    token = esc(event["token"]) or "&nbsp;"
                    sampling_chips.append(
                        f'<span class="token-chip" style="background:{bg};color:{fg}" '
                        f'title="{prob * 100:.1f}% confidence">{token}</span>'
                    )
                    sampling_block = (
                        f'<div class="stage-block">'
                        f'<span class="stage-label">1. Sampling</span>'
                        f'<span class="stage-math">{_MATH["generate"]}</span>'
                        f"</div>"
                        f'<div class="token-row">{"".join(sampling_chips)}</div>'
                    )
                    if len(blocks) == 0 or "1. Sampling" in blocks[0]:
                        blocks[:1] = [sampling_block]
                    else:
                        blocks.insert(0, sampling_block)
                    status_line.empty()
                    flush()
                elif stage == "generate" and "sa_pairs" in event:
                    status_line.empty()
                    pairs = sorted(event["sa_pairs"], key=lambda p: p["step"])
                    chips: list[str] = []
                    for pair in pairs:
                        prob = math.exp(pair["logprob"])
                        bg = seq_step(prob)
                        fg = text_color_for(bg)
                        token = esc(pair["action_token"]) or "&nbsp;"
                        chips.append(
                            f'<span class="token-chip" style="background:{bg};color:{fg}" '
                            f'title="{prob * 100:.1f}% confidence">{token}</span>'
                        )
                    completion = esc(event["generated_texts"][0])
                    cur["completion"] = completion
                    sampling_block = (
                        f'<div class="stage-block">'
                        f'<span class="stage-label">1. Sampling</span>'
                        f'<span class="stage-math">{_MATH["generate"]}</span>'
                        f'<span class="stage-completion">{completion}</span>'
                        f"</div>"
                        f'<div class="token-row">{"".join(chips)}</div>'
                    )
                    if len(blocks) > 0 and "1. Sampling" in blocks[0]:
                        blocks[0] = sampling_block
                    else:
                        blocks.append(sampling_block)
                    sampling_chips = []
                    flush()
                elif stage == "heartbeat":
                    if last_load:
                        status_line.markdown(f"⏳ {last_load}  —  {event['elapsed']}s")
                elif stage == "done":
                    if cur:
                        step_records.append(cur)
                    st.session_state["rlhf-ppo_ppo_data"] = event
                    st.session_state["rlhf-ppo_step_records"] = step_records
                    status_line.empty()
                elif stage == "error":
                    status_line.empty()
                    st.error(f"PPO step failed: {event['detail']}")
                elif stage == "rewards" and "rewards" in event:
                    cur["reward"] = event["rewards"][0]
                    vals = ", ".join(f"{r:.3f}" for r in event["rewards"])
                    blocks.append(
                        f'<div class="stage-block">'
                        f'<span class="stage-label">3. Reward Scoring</span>'
                        f'<span class="stage-math">{_MATH["rewards"]}</span>'
                        f"</div>"
                        f'<div class="stage-scalar">{vals}</div>'
                    )
                    flush()
                elif stage in {"value_loss", "policy_loss"}:
                    label = (
                        "9. Value Function Loss" if stage == "value_loss" else "11. PPO-Clip Loss"
                    )
                    math_key = "value_loss" if stage == "value_loss" else "policy_loss"
                    value = (
                        event.get("value_loss")
                        if stage == "value_loss"
                        else event.get("policy_loss")
                    )
                    if value is not None:
                        if stage == "value_loss":
                            cur["value_loss"] = value
                        else:
                            cur["policy_loss"] = value
                        blocks.append(
                            f'<div class="stage-block">'
                            f'<span class="stage-label">{label}</span>'
                            f'<span class="stage-math">{_MATH[math_key]}</span>'
                            f"</div>"
                            f'<div class="stage-scalar">{value:.4f}</div>'
                        )
                        flush()
                elif stage == "kl_divergence" and "sa_pairs" in event:
                    kls = [p["kl_div"] for p in event["sa_pairs"]]
                    cur["avg_kl"] = sum(kls) / len(kls) if kls else 0.0
                    label, field = _rendering[stage]
                    block_html = _render_heatmap(label, stage, field, event, _MATH)
                    blocks.append(block_html)
                    flush()
                elif stage == "policy_ratio" and "sa_pairs" in event:
                    ratios = [p["policy_ratio"] for p in event["sa_pairs"]]
                    cur["avg_ratio"] = sum(ratios) / len(ratios) if ratios else 1.0
                    label, field = _rendering[stage]
                    block_html = _render_heatmap(label, stage, field, event, _MATH)
                    blocks.append(block_html)
                    # Per-step summary line.
                    step_reward = cur.get("reward", float("nan"))
                    vl = cur.get("value_loss", float("nan"))
                    pl = cur.get("policy_loss", float("nan"))
                    summary = '<div class="step-summary">'
                    summary += "<table><tr>"
                    summary += f"<td>reward</td><td>{step_reward:.2f}</td>"
                    summary += f"<td>value loss</td><td>{vl:.1f}</td>"
                    summary += f"<td>policy loss</td><td>{pl:+.3f}</td>"
                    summary += f"<td>avg KL</td><td>{cur['avg_kl']:.2f}</td>"
                    summary += f"<td>avg ratio</td><td>{cur['avg_ratio']:.3f}</td>"
                    summary += "</tr></table></div>"
                    blocks.append(summary)
                    flush()
                elif stage in _rendering and "sa_pairs" in event:
                    label, field = _rendering[stage]
                    block_html = _render_heatmap(label, stage, field, event, _MATH)
                    blocks.append(block_html)
                    flush()
        except requests.exceptions.RequestException as error:
            show_stream_error(error)

    ppo_data = st.session_state.get("rlhf-ppo_ppo_data")
    if ppo_data:
        records = st.session_state.get("rlhf-ppo_step_records", [])
        if records and len(records) > 1:
            r_first = records[0].get("reward", float("nan"))
            r_last = records[-1].get("reward", float("nan"))
            vl_first = records[0].get("value_loss", float("nan"))
            vl_last = records[-1].get("value_loss", float("nan"))
            pl_first = records[0].get("policy_loss", float("nan"))
            pl_last = records[-1].get("policy_loss", float("nan"))
            kl_first = records[0].get("avg_kl", float("nan"))
            kl_last = records[-1].get("avg_kl", float("nan"))

            def _arrow(a: float, b: float) -> str:
                if math.isnan(a) or math.isnan(b) or abs(b - a) < 0.01:
                    return " →"
                return " ↑" if b > a else " ↓"

            st.markdown(
                f'<div class="summary-card">'
                f"<h4>{len(records)} PPO steps — before → after</h4>"
                f'<table class="summary-table">'
                f'<tr><th style="text-align:left;font-weight:500;color:var(--text-muted)">metric</th>'
                f'<th style="text-align:right;font-weight:500;color:var(--text-muted)">step 1</th>'
                f'<th style="text-align:right;font-weight:500;color:var(--text-muted)">step {len(records)}</th></tr>'
                f"<tr><td>Reward</td>"
                f'<td style="text-align:right">{r_first:.2f}</td>'
                f'<td style="text-align:right">{r_last:.2f}'
                f'<span class="{"summary-good" if r_last > r_first else "summary-bad" if r_last < r_first else ""}">{_arrow(r_first, r_last)}</span></td></tr>'
                f"<tr><td>Value loss</td>"
                f'<td style="text-align:right">{vl_first:.1f}</td>'
                f'<td style="text-align:right">{vl_last:.1f}'
                f'<span class="summary-{"good" if vl_last < vl_first else "bad" if vl_last > vl_first else ""}">{_arrow(vl_first, vl_last)}</span></td></tr>'
                f"<tr><td>Policy loss</td>"
                f'<td style="text-align:right">{pl_first:+.3f}</td>'
                f'<td style="text-align:right">{pl_last:+.3f}</td></tr>'
                f"<tr><td>KL divergence</td>"
                f'<td style="text-align:right">{kl_first:.2f}</td>'
                f'<td style="text-align:right">{kl_last:.2f}'
                f'<span class="">{_arrow(kl_first, kl_last)}</span></td></tr>'
                f"</table></div>",
                unsafe_allow_html=True,
            )
        st.info(ppo_data["note"])
        with st.expander("Raw response"):
            st.json(ppo_data)

# --- Write-up + diagrams + stats below the demo, so users can try it immediately ---

render_service_details(_PAGE)
