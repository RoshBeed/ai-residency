"""Vector search: embed, index in Redis, retrieve."""

import requests
import streamlit as st

from client import stream
from components import render_stats, render_word_trace
from shared import esc, service_page, show_stream_error

SERVICE = "vector-search-retrieval"


def render() -> None:
    with service_page(SERVICE):
        query = st.text_input("Search query", value="was ronald reagan a democrat")
        top_k = st.slider("Top K", 1, 10, 5)
        if st.button("Search", type="primary") and query:
            st.caption("Live -- each stage below arrives the moment it completes:")
            tracker = st.empty()
            live_words = st.empty()
            retrieved_lines: list[str] = []
            try:
                for event in stream(
                    SERVICE,
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
                        retrieved_lines[-1] = (
                            f'<span class="done">✓</span> {n_res} results retrieved'
                        )
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
                show_stream_error(error, SERVICE)
