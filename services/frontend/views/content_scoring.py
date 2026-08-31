"""Content scoring: try the claims in the write-up against the running model.

Each tab answers a question the write-up raises, so a reader can move between the two:
what does this post score, what actually moves that number, and do the four
architectures agree.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import streamlit as st

from client import call
from components import render_probability_bars, render_stats, render_word_trace
from shared import service_page, show

SERVICE = "content-scoring"

ARCHITECTURES = {
    "text_only": "Title only",
    "features_only": "Metadata only",
    "early": "Early fusion",
    "late": "Late fusion",
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
    """Variants against the post as typed, ordered by predicted score."""
    ordered = sorted(rows, key=lambda row: -row[1])
    if not ordered:
        return
    largest = max(value for _, value in ordered) or 1.0
    st.markdown(
        render_probability_bars({name: value / largest for name, value in ordered}, ordered[0][0]),
        unsafe_allow_html=True,
    )
    for name, value in ordered:
        change = value - baseline
        arrow = "▲" if change > 0.05 else ("▼" if change < -0.05 else "—")
        st.markdown(f"`{value:>6.1f}`  {arrow} {change:+.1f} &nbsp; {name}", unsafe_allow_html=True)


def _render_score(title: str, url: str, author: str, when: datetime) -> None:
    """Tab one: what does this post score, and how sure is that."""
    data = show(
        call(
            SERVICE,
            "/predict",
            {"title": title, "url": url, "author": author, "posted_at": when.isoformat()},
        )
    )
    if not data:
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


def _render_what_moves_it(title: str, url: str, author: str, when: datetime) -> None:
    """Tab two: change one thing at a time and watch the prediction move."""
    baseline = _score(title, url, author, when)
    if baseline is None:
        st.info("The backend did not answer.")
        return
    base = float(baseline["predicted_score"])
    st.markdown(f"Your post as typed scores **{base:.1f} upvotes**. Now change one thing:")

    st.markdown("**Where it links**")
    _delta_table(
        [
            (name, float((_score(title, swap, author, when) or {}).get("predicted_score", 0.0)))
            for name, swap in DOMAIN_SWAPS.items()
        ],
        base,
    )

    st.markdown("**When it is posted**")
    _delta_table(
        [
            (
                name,
                float(
                    (_score(title, url, author, when.replace(hour=hour)) or {}).get(
                        "predicted_score", 0.0
                    )
                ),
            )
            for name, hour in HOUR_SWAPS.items()
        ],
        base,
    )

    st.markdown("**How the title opens**")
    stripped = title
    for prefix in ("Show HN: ", "Ask HN: ", "Tell HN: "):
        stripped = stripped.replace(prefix, "")
    _delta_table(
        [
            (
                f"{prefix or 'no prefix'}{stripped[:34]}…",
                float(
                    (_score(prefix + stripped, url, author, when) or {}).get("predicted_score", 0.0)
                ),
            )
            for prefix in PREFIX_SWAPS
        ],
        base,
    )

    st.caption(
        "Across the whole test set, removing the timestamp branch costs the model more accuracy "
        "than removing the title branch. That is not the same as what you will see here: on any "
        "single post the domain usually swings the number hardest, because it is one categorical "
        "the model knows a lot about, while the timestamp earns its keep across every post at once. "
        "Both are metadata, and both routinely move the prediction more than rewording the title."
    )


def _render_architectures(title: str, url: str, author: str, when: datetime) -> None:
    """Tab three: the four models on one post."""
    data = show(
        call(
            SERVICE,
            "/compare",
            {"title": title, "url": url, "author": author, "posted_at": when.isoformat()},
        )
    )
    if not data or not data["predictions"]:
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
    st.caption(
        "Title only and metadata only are single MLPs — the ablations that make “combining helps” "
        "testable rather than assumed. Early fusion concatenates both and uses one network; late "
        "fusion gives each input its own and joins them at the end. Across the test set the two "
        "fusion models are separated by less than the difference between two training runs, so the "
        "write-up does not call a winner between them."
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

        when = (datetime.now(UTC) - timedelta(days=1)).replace(
            hour=hour, minute=0, second=0, microsecond=0
        )

        score_tab, moves_tab, models_tab = st.tabs(
            ["Score it", "What moves the number", "Four architectures"]
        )
        with score_tab:
            if st.button("Predict", type="primary", key="predict") and title:
                _render_score(title, url, author, when)
        with moves_tab:
            st.markdown("Hold the title fixed and change one thing at a time.")
            if st.button("Run the comparison", type="primary", key="moves") and title:
                _render_what_moves_it(title, url, author, when)
        with models_tab:
            st.markdown("The same post through all four models.")
            if st.button("Compare", type="primary", key="compare") and title:
                _render_architectures(title, url, author, when)
