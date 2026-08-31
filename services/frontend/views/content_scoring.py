"""Content scoring: four fusion architectures over one Hacker News post."""

from typing import Any

import requests
import streamlit as st

from client import call, stream
from components import render_probability_bars, render_stats, render_word_trace
from shared import service_page, show, show_stream_error

SERVICE = "content-scoring"

_ARCHITECTURE_LABELS = {
    "text_only": "Text only",
    "features_only": "Features only",
    "early": "Early fusion",
    "late": "Late fusion",
}

_EXAMPLES = {
    "Show HN (GitHub)": (
        "Show HN: I built a CLI tool that writes your commit messages",
        "https://github.com/example/commit-ai",
    ),
    "Research paper": ("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
    "Ask HN (text post)": ("Ask HN: What are you working on this weekend?", ""),
    "Clickbait": ("10 JavaScript tricks you won't believe", "https://medium.com/@someone/x"),
}


def _comparison_bars(predictions: dict[str, float]) -> str:
    """The four architectures' predictions as bars, scaled to the largest."""
    largest = max(predictions.values()) if predictions else 1.0
    scaled = {
        _ARCHITECTURE_LABELS.get(name, name): (value / largest if largest > 0 else 0.0)
        for name, value in predictions.items()
    }
    best = max(scaled, key=lambda name: scaled[name]) if scaled else None
    return render_probability_bars(scaled, best)


def _render_comparison(title: str, url: str, author: str) -> None:
    """Ask every architecture the same question and show the four answers together."""
    data = show(call(SERVICE, "/compare", {"title": title, "url": url, "author": author}))
    if not data:
        return

    predictions: dict[str, float] = data["predictions"]
    if not predictions:
        st.info("No architectures are loaded yet.")
        return

    st.markdown(
        '<div class="trace-caption">Each architecture sees exactly the same post. '
        "They differ only in where the title meets the metadata.</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(len(predictions))
    for column, (name, value) in zip(columns, predictions.items(), strict=True):
        with column:
            st.metric(_ARCHITECTURE_LABELS.get(name, name), f"{value:.1f}", help="expected upvotes")

    st.markdown(_comparison_bars(predictions), unsafe_allow_html=True)

    spread = max(predictions.values()) - min(predictions.values())
    st.markdown(
        render_stats(
            [
                ("Domain", data["domain"]),
                ("Architectures", str(len(predictions))),
                ("Spread", f"{spread:.1f} upvotes"),
            ]
        ),
        unsafe_allow_html=True,
    )
    if data.get("missing"):
        st.caption(
            "Not loaded: "
            + ", ".join(_ARCHITECTURE_LABELS.get(name, name) for name in data["missing"])
        )
    with st.expander("Raw response"):
        st.json(data)


def _render_stream(title: str, url: str, author: str) -> None:
    """The served architecture, one pipeline stage at a time."""
    st.caption("Live -- each stage below arrives the moment it completes:")
    tracker = st.empty()
    live_words = st.empty()
    done_lines: list[str] = []

    def paint() -> None:
        tracker.markdown(
            f'<div class="tracker">{"<br>".join(done_lines)}</div>', unsafe_allow_html=True
        )

    try:
        for event in stream(
            SERVICE,
            "/predict/stream",
            {"title": title, "url": url, "author": author},
            timeout=15,
        ):
            stage = event["stage"]
            if stage == "word_lookup":
                word_trace = event["word_trace"]
                known_c = sum(1 for w in word_trace if w["known"])
                done_lines.append(
                    f'<span class="done">✓</span> Word lookup: {known_c}/{len(word_trace)} recognised'
                )
                done_lines.append('<span class="active">⟳</span> Building features…')
                paint()
                live_words.markdown(render_word_trace(word_trace), unsafe_allow_html=True)
            elif stage == "features":
                domain = event["domain"]
                done_lines[-1] = (
                    f'<span class="done">✓</span> Features built '
                    f"(domain {domain}, {'linked' if event['has_url'] else 'text post'})"
                )
                done_lines.append('<span class="active">⟳</span> Fusing…')
                paint()
            elif stage == "fusion":
                label = _ARCHITECTURE_LABELS.get(event["fusion"], event["fusion"])
                done_lines[-1] = f'<span class="done">✓</span> {label}'
                done_lines.append('<span class="active">⟳</span> Predicting score…')
                paint()
            elif stage == "done":
                data: dict[str, Any] = event
                done_lines[-1] = (
                    f'<span class="done">✓</span> Predicted {data["predicted_score"]:.1f} upvotes'
                )
                paint()
                st.markdown(
                    '<div class="trace-caption">Words looked up in the extended vocabulary '
                    "— text8 plus the Hacker News title corpus. Unknown words are "
                    "skipped.</div>",
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
                            ("Domain", data["domain"] or "text post"),
                        ]
                    ),
                    unsafe_allow_html=True,
                )
                left_metric, right_metric = st.columns(2)
                with left_metric:
                    st.metric("Expected score", f"{data['predicted_score']:.1f} upvotes")
                with right_metric:
                    st.metric("Chance of 100+", f"{data.get('viral_probability', 0.0):.1%}")
                st.caption(
                    "Hacker News is close to a lottery — the same URL posted twice scores "
                    "similarly only about a quarter of the time. The expected count is the "
                    "middle of a very wide distribution, not a forecast."
                )
                with st.expander("Raw response"):
                    st.json(data)
            elif stage == "error":
                tracker.empty()
                st.error(f"Prediction failed: {event['detail']}")
    except requests.exceptions.RequestException as error:
        show_stream_error(error, SERVICE)


def render() -> None:
    with service_page(SERVICE):
        example = st.selectbox("Start from an example", list(_EXAMPLES), index=0)
        default_title, default_url = _EXAMPLES[example]

        title = st.text_input("Hacker News post title", value=default_title)
        left, right = st.columns(2)
        with left:
            url = st.text_input("URL (blank for a text post)", value=default_url)
        with right:
            author = st.text_input("Author (optional)", value="")

        compare_tab, stream_tab = st.tabs(["Compare architectures", "Watch the pipeline"])
        with compare_tab:
            st.markdown(
                "**All four at once** — text only, features only, early fusion and late "
                "fusion, scoring the same post."
            )
            if st.button("Compare", type="primary", key="compare") and title:
                _render_comparison(title, url, author)
        with stream_tab:
            st.markdown("**The served architecture**, stage by stage.")
            if st.button("Predict score", type="primary", key="predict") and title:
                _render_stream(title, url, author)
