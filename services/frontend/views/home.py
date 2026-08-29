"""The gallery: one card per system, and the way into each of them.

Shown only when more than one page is configured. With `AI_RESIDENCY_PAGES` naming a
single system there is no gallery worth landing on, so that system *is* the landing
page -- which is what lets the one hub image also be the UI for one service.
"""

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from projects import PAGES, PROJECTS, tag_link


def render(targets: dict[str, StreamlitPage]) -> None:
    st.markdown(
        "<h2 style='text-align:center;margin:0 0 2px;font-size:1.2rem;'>ML Systems Portfolio</h2>"
        "<hr style='margin:0 0 6px;border:none;border-top:1px solid var(--border);'>",
        unsafe_allow_html=True,
    )

    # Project cards — the main focus, above the fold.
    service_keys = list(PAGES)
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
                    st.switch_page(targets[wk])

    st.markdown(
        "<p style='text-align:center;font-size:0.85rem;color:var(--text-muted);margin-top:8px;'>"
        "End-to-end ML pipelines spanning NLP, computer vision, speech, and reinforcement learning."
        "</p>",
        unsafe_allow_html=True,
    )
