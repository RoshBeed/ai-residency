"""Content scoring: a two-tower relevance model."""

import requests
import streamlit as st

from client import stream
from components import render_stats, render_word_trace
from shared import service_page, show_stream_error

SERVICE = "content-scoring"


def render() -> None:
    with service_page(SERVICE):
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
                    SERVICE,
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
                        done_lines.append(
                            '<span class="active">⟳</span> Building embedding vector…'
                        )
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
                            "Words looked up in the CBOW vocabulary (71,290 words). "
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
                show_stream_error(error, SERVICE)
