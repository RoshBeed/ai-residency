"""Word embeddings: nearest neighbours and analogies."""

import streamlit as st

from client import call
from shared import service_page, show

SERVICE = "word-embeddings"


def render() -> None:
    with service_page(SERVICE):
        st.markdown(
            "**Nearest neighbours** — the words closest to a query word in the learned space."
        )
        word = st.text_input("Word", value="python")
        if st.button("Find nearest", type="primary") and word:
            data = show(call(SERVICE, "/nearest", {"word": word}))
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
            data = show(call(SERVICE, "/analogy", {"a": a, "b": b, "c": c}))
            if data:
                for neighbour, similarity in data["nearest"]:
                    st.markdown(
                        f"`{a} − {b} + {c}` ≈ **{neighbour}** "
                        f'<span class="dim">(cosine {similarity:.3f})</span>',
                        unsafe_allow_html=True,
                    )
