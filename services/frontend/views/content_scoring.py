"""Content scoring: try the claims in the write-up against the running model.

Each tab answers a question the write-up raises, so a reader can move between the two:
what does this post score, what actually moves that number, and do the four
architectures agree.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import streamlit as st

from client import call
from components import (
    render_delta_bars,
    render_probability_bars,
    render_stats,
    render_word_trace,
)
from shared import service_page

SERVICE = "content-scoring"

ARCHITECTURES = {
    "text_only": "Title only",
    "features_only": "Metadata only",
    "early": "Early fusion",
    "block_diagonal": "Block-diagonal",
    "late": "Late fusion",
}

WHAT_EACH_IS = {
    "text_only": "the pooled title, nothing else — an ablation",
    "features_only": "the metadata, no text — the other ablation",
    "early": "one dense layer over the title and metadata together",
    "block_diagonal": "the same layer with the cross-group weights removed",
    "late": "a full-width tower per group, combined by a second layer",
}

EXAMPLES = {
    "Show HN on GitHub": (
        "Show HN: I built a database that runs entirely in the browser",
        "https://github.com/example/browser-db",
    ),
    "Research paper": ("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
    "Ask HN (no link)": ("Ask HN: What are you working on this weekend?", ""),
    "Clickbait": ("10 JavaScript tricks you won't believe", "https://medium.com/@someone/x"),
    "Breaking news": ("Google lays off 12,000 employees", "https://nytimes.com/tech/layoffs"),
}

# The write-up's ablation says the timestamp outweighs the title. These swaps let a
# reader check that on a post they chose themselves, rather than take it on trust from a
# bar chart drawn over 69,275 posts they cannot see.
DOMAIN_SWAPS = {
    "github.com": "https://github.com/x/y",
    "arxiv.org": "https://arxiv.org/abs/1706.03762",
    "nytimes.com": "https://nytimes.com/x",
    "medium.com": "https://medium.com/@a/b",
    "a blog nobody knows": "https://some-personal-site.dev/post",
    "no link (text post)": "",
}
HOUR_SWAPS = {"03:00 UTC": 3, "09:00 UTC": 9, "15:00 UTC": 15, "21:00 UTC": 21}
PREFIX_SWAPS = ("Show HN: ", "Ask HN: ", "")


@st.cache_data(ttl=300, show_spinner=False)
def _everything(title: str, url: str, author: str, moment: str) -> dict[str, Any]:
    """Every call the three tabs need, made once.

    Args:
        title: The post title.
        url: Where it links, or empty.
        author: The submitter, or empty.
        moment: When it was posted, ISO-8601. A string rather than a datetime so the
            cache key is hashable.

    Returns:
        `{"score", "compare", "domains", "hours", "prefixes"}`.

    Cached and computed up front rather than behind a button per tab. Three tabs each
    with its own button means three clicks to see one post, and a Streamlit button is
    False on the next rerun -- so changing the hour would blank the results the previous
    click produced. With this, switching tabs is instant and every input change refreshes
    all three views together.

    The TTL matters during development: `make local-up` restarts the container on every
    edit, and a permanent cache would keep serving answers from the model that was
    running before the change.
    """
    when = datetime.fromisoformat(moment)
    base = _score(title, url, author, when)
    stripped = title
    for prefix in ("Show HN: ", "Ask HN: ", "Tell HN: "):
        stripped = stripped.replace(prefix, "")
    return {
        "score": base,
        "compare": (
            call(
                SERVICE,
                "/compare",
                {"title": title, "url": url, "author": author, "posted_at": moment},
            ).data
        ),
        "domains": [
            (name, _predicted(_score(title, swap, author, when)))
            for name, swap in DOMAIN_SWAPS.items()
        ],
        "hours": [
            (name, _predicted(_score(title, url, author, when.replace(hour=hour))))
            for name, hour in HOUR_SWAPS.items()
        ],
        "prefixes": [
            (
                f"{prefix or 'no prefix'}{stripped[:34]}…",
                _predicted(_score(prefix + stripped, url, author, when)),
            )
            for prefix in PREFIX_SWAPS
        ],
    }


def _predicted(data: dict[str, Any] | None) -> float:
    """The expected count from a response, or zero if the backend refused it."""
    return float((data or {}).get("predicted_score", 0.0))


def _score(
    title: str, url: str, author: str, when: datetime | None = None
) -> dict[str, Any] | None:
    """One prediction, or None if the backend refused it."""
    payload: dict[str, Any] = {"title": title, "url": url, "author": author}
    if when is not None:
        payload["posted_at"] = when.isoformat()
    result = call(SERVICE, "/predict", payload)
    return result.data if result.ok else None


def _delta_table(rows: list[tuple[str, float]], baseline: float) -> None:
    """Variants against the post as typed, ordered by predicted score.

    One line each. The bar, the predicted count and the change from the post as typed
    share a row rather than the numbers being repeated under the chart, which is what
    fits all three comparisons on a screen instead of scrolling past thirty rows.
    """
    ordered = sorted(rows, key=lambda row: -row[1])
    if not ordered:
        return
    st.markdown(render_delta_bars(ordered, baseline), unsafe_allow_html=True)


def _render_score(data: dict[str, Any] | None) -> None:
    """Tab one: what does this post score, and how sure is that."""
    if not data:
        st.info("The backend did not answer.")
        return

    left, right = st.columns(2)
    with left:
        st.metric("Expected score", f"{data['predicted_score']:.1f} upvotes")
    with right:
        st.metric("Chance of reaching 100+", f"{data.get('viral_probability', 0.0):.1%}")

    st.caption(
        "The expected score is the middle of a very wide distribution, not a forecast. The same "
        "link posted to Hacker News twice scores similarly only about a quarter of the time, so "
        "the probability on the right is the more honest half of the answer."
    )

    known = sum(1 for word in data["word_trace"] if word["known"])
    st.markdown(
        render_stats(
            [
                ("Words recognised", f"{known} / {len(data['word_trace'])}"),
                ("Domain", data["domain"] or "text post"),
                ("Architecture", ARCHITECTURES.get(data["fusion"], data["fusion"])),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="trace-caption">Looked up in the extended vocabulary — text8 plus 710,000 '
        "Hacker News titles. Unknown words are skipped.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(render_word_trace(data["word_trace"]), unsafe_allow_html=True)


def _render_what_moves_it(bundle: dict[str, Any]) -> None:
    """Tab two: change one thing at a time and watch the prediction move."""
    base = _predicted(bundle["score"])
    if base == 0.0:
        st.info("The backend did not answer.")
        return
    st.markdown(f"Your post as typed scores **{base:.1f} upvotes**. Now change one thing:")

    # Side by side rather than stacked: the three questions are independent, so reading
    # them in parallel is the point, and stacking them pushed the last one off-screen.
    for column, (heading, key) in zip(
        st.columns(3),
        (
            ("Where it links", "domains"),
            ("When it is posted", "hours"),
            ("How the title opens", "prefixes"),
        ),
        strict=True,
    ):
        with column:
            st.markdown(f"**{heading}**")
            _delta_table(bundle[key], base)

    st.caption(
        "Across the whole test set, removing the timestamp branch costs the model more accuracy "
        "than removing the title branch. That is not the same as what you will see here: on any "
        "single post the domain usually swings the number hardest, because it is one categorical "
        "the model knows a lot about, while the timestamp earns its keep across every post at once. "
        "Both are metadata, and both routinely move the prediction more than rewording the title."
    )


def _render_architectures(data: dict[str, Any] | None) -> None:
    """Tab three: the four models on one post."""
    if not data or not data.get("predictions"):
        st.info("The backend did not answer.")
        return

    predictions: dict[str, float] = data["predictions"]
    columns = st.columns(len(predictions))
    for column, (name, value) in zip(columns, predictions.items(), strict=True):
        with column:
            st.metric(ARCHITECTURES.get(name, name), f"{value:.1f}")

    largest = max(predictions.values()) or 1.0
    best = max(predictions, key=lambda key: predictions[key])
    st.markdown(
        render_probability_bars(
            {ARCHITECTURES.get(n, n): v / largest for n, v in predictions.items()},
            ARCHITECTURES.get(best, best),
        ),
        unsafe_allow_html=True,
    )
    details = data.get("details") or {}
    if details:
        st.markdown("**What each one is**")
        rows = [
            "| model | what it is | first layer | then | parameters |",
            "|---|---|---|---|---|",
        ]
        for name in predictions:
            spec = details.get(name, {})
            # Escaped: late fusion's shape is five blocks joined by a pipe, which is also
            # markdown's cell delimiter, so it split one column into six and the row
            # collapsed. Escaping here rather than changing the separator keeps the API
            # returning data and the renderer responsible for its own syntax.
            shape = str(spec.get("first_layer", "—")).replace("|", "\\|")
            shared = str(spec.get("shared_layers", "—"))
            rows.append(
                f"| {ARCHITECTURES.get(name, name)} | {WHAT_EACH_IS.get(name, '')} "
                f"| `{shape}` | `{shared} → 1` "
                f"| {spec.get('parameters', 0):,} |"
            )
        st.markdown("\n".join(rows))

    chosen = st.selectbox(
        "Show the wiring for", list(predictions), format_func=lambda n: ARCHITECTURES.get(n, n)
    )
    graph = (details.get(chosen) or {}).get("graph")
    if graph:
        st.graphviz_chart(graph, width="stretch")
        st.caption(
            f"Generated from the running {ARCHITECTURES.get(chosen, chosen)} model. "
            "`make -C services/content-scoring onnx` writes the full computation graph "
            "for [Netron](https://netron.app) if you want every operator."
        )

    st.caption(
        "Read off the running models, not written down beside them. **Early fusion and "
        "block-diagonal are the honest pair**: identical input, hidden width, depth and head, "
        "differing only in whether the title and the metadata are wired to each other. "
        "Block-diagonal is the smaller of the two because it *is* early fusion with weights "
        "deleted, so the difference between them is the value of that crosstalk and nothing else. "
        "Late fusion is not that comparison — it carries five times the hidden units and an extra "
        "layer, so a gap against early could be connectivity, width or depth."
    )
    if data.get("missing"):
        st.caption("Not loaded: " + ", ".join(ARCHITECTURES.get(n, n) for n in data["missing"]))


def render() -> None:
    with service_page(SERVICE):
        example = st.selectbox("Start from an example", list(EXAMPLES), index=0)
        default_title, default_url = EXAMPLES[example]

        title = st.text_input("Hacker News title", value=default_title)
        left, middle, right = st.columns([2, 1, 1])
        with left:
            url = st.text_input("URL (blank for a text post)", value=default_url)
        with middle:
            hour = st.slider("Hour posted (UTC)", 0, 23, 15)
        with right:
            author = st.text_input("Author", value="")

        if not title:
            st.info("Type a title to score it.")
            return

        when = (datetime.now(UTC) - timedelta(days=1)).replace(
            hour=hour, minute=0, second=0, microsecond=0
        )
        with st.spinner("Scoring…"):
            bundle = _everything(title, url, author, when.isoformat())

        score_tab, moves_tab, models_tab = st.tabs(
            ["Score it", "What moves the number", "Four architectures"]
        )
        with score_tab:
            _render_score(bundle["score"])
        with moves_tab:
            st.markdown("The same title, with one thing changed at a time.")
            _render_what_moves_it(bundle)
        with models_tab:
            st.markdown("The same post through all four models.")
            _render_architectures(bundle["compare"])
