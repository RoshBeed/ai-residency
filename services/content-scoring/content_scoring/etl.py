"""Hacker News stories: fetch, clean, split.

The service used to train on 782 posts cleaned out of the course's Postgres mirror.
That host is unreachable and the slice it served is gone, so the extract could not be
rebuilt -- the config said as much. This module replaces it with a corpus of ~870k
posts and a path that runs end to end on any machine with no credentials.

Three sources, one record shape:

* `mirror` (default) -- `roshbeed/ai-residency-hn-posts`, a public dataset repo. This
  is what a normal run uses.
* `api` -- `hacker-news.firebaseio.com`, the real upstream, no authentication. Bounded
  by `--max-items` and used to *verify* the mirror rather than to replace it: crawling
  870k items one at a time takes hours, and the point of having it is that the corpus
  is re-derivable, not that it is re-derived nightly.
* `upstream` -- copies `julien040/hacker-news-posts` into the mirror. Run once, by
  hand; the only place in the data path that needs a token.

**Only the raw mirror is published.** Cleaning and splitting are deterministic code
over an input that never changes, so the splits are reproducible without being
uploaded. A cleaned copy on the Hub would be a second artifact that can drift from the
code that produces it, for no reproducibility gain. `data/` is a local cache.
"""

import argparse
import html
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlparse

import pandas as pd
import requests

from ai_residency_common import get_logger, push_checkpoint, resolve_checkpoint
from config import Config

__all__ = [
    "SplitSizes",
    "build_splits",
    "clean_frame",
    "fetch_from_api",
    "load_raw",
    "normalise_text",
    "registered_domain",
    "split_by_time",
    "tokenise",
]

logger = get_logger(__name__)

_API_ROOT = "https://hacker-news.firebaseio.com/v0"
_COLUMNS = ["id", "title", "url", "score", "time", "author", "comments"]

# Characters that actually turn up in Hacker News titles, mapped to ASCII. The version
# this replaces had ~120 entries reaching as far as "does not double turnstile", which
# no title has ever contained; everything unmapped is dropped by the ASCII pass below,
# so a longer table buys nothing but maintenance. Smart quotes and en/em dashes are the
# ones that matter -- they are pervasive, and dropping them silently welds words
# together ("don t" is recoverable, "dont" is not).
_REPLACEMENTS = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    "→": "->",
    "←": "<-",
    "±": "+/-",
    "×": "x",
    "÷": "/",
    "©": "(c)",
    "®": "(r)",
    "™": "(tm)",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "°": " degrees",
    "µ": "u",
    "≠": "!=",
    "≤": "<=",
    "≥": ">=",
    "≈": "~=",
    "∞": "inf",
    "√": "sqrt",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-+#][a-z0-9]+)*[+#]*")
"""Keeps `c++`, `gpt-4`, `node.js`, `f#` and `k8s` whole.

The trailing `[+#]*` is what saves `c++` and `f#`: without it the pattern requires
an alphanumeric after every `+`, so `c++` tokenised to `c` -- silently, and only for
the language names most likely to matter on this corpus.

text8 is lowercase alphabetic, so the pretrained vocabulary contains none of these --
which is the entire reason this service extends it. Splitting on every non-letter would
throw away the vocabulary the extension exists to add.
"""


@dataclass(frozen=True)
class SplitSizes:
    """Row counts per split, reported so the README quotes a measurement."""

    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        """Rows across all three splits."""
        return self.train + self.val + self.test


def normalise_text(text: str) -> str:
    """Fold a title to ASCII without silently deleting the characters that carry words.

    Args:
        text: A raw title, possibly carrying HTML entities and Unicode punctuation.

    Returns:
        The ASCII form. HTML entities are unescaped first -- Hacker News serves
        `&#x27;` and `&amp;` in titles -- then the replacement table runs, and only
        then is anything still non-ASCII dropped.

    The order is the whole point. The implementation this replaces stripped to ASCII
    *before* consulting its replacement table, so every entry in the table was dead
    code and `cost: EUR20 - GBP15` came out as `cost: 20  15`.
    """
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    for source, target in _REPLACEMENTS.items():
        text = text.replace(source, target)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def tokenise(title: str) -> list[str]:
    """Lowercase a title into vocabulary tokens.

    Args:
        title: A normalised title.

    Returns:
        Lowercased tokens, keeping internal `.`, `-`, `+` and `#` so that `gpt-4`,
        `c++`, `node.js` and `f#` survive as single words.
    """
    return _TOKEN_RE.findall(title.lower())


def registered_domain(url: str | None) -> str:
    """The domain a post links to, as a categorical value.

    Args:
        url: The post's URL, or None/empty for a text post.

    Returns:
        The lowercased host with a leading `www.` removed, or `<none>` for a text post
        (`Ask HN`, and anything else self-posted). `<none>` is a real category rather
        than a missing value: 5.6% of posts are text posts and they behave differently
        from linked ones, so collapsing them into an unknown bucket would throw away a
        signal rather than handle an absence.
    """
    if not url:
        return "<none>"
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "<none>"


def load_raw(config: Config) -> pd.DataFrame:
    """Load the raw mirror, pulling it from the Hub on first use.

    Args:
        config: Supplies `raw_posts_file` and `hn_repo_id`.

    Returns:
        The raw stories frame.

    Raises:
        FileNotFoundError: If the mirror is neither local nor fetchable.
    """
    path, source = resolve_checkpoint(config.raw_posts_file, config.hn_repo_id, repo_type="dataset")
    if path is None:
        raise FileNotFoundError(
            f"The Hacker News mirror is not at {config.raw_posts_file} and could not be "
            f"fetched from {config.hn_repo_id}."
        )
    frame = pd.read_parquet(path)
    logger.info("Loaded %d raw stories from %s (%s)", len(frame), path, source)
    return frame


def _fetch_item(item_id: int, session: requests.Session, timeout: int) -> dict[str, object] | None:
    """Fetch one Hacker News item, returning None for anything that is not a story."""
    try:
        response = session.get(f"{_API_ROOT}/item/{item_id}.json", timeout=timeout)
        response.raise_for_status()
        item = response.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(item, dict) or item.get("type") != "story" or item.get("deleted"):
        return None
    if not item.get("title") or item.get("score") is None:
        return None
    return {
        "id": item["id"],
        "title": item["title"],
        "url": item.get("url") or "",
        "score": item["score"],
        "time": item.get("time", 0),
        "author": item.get("by", ""),
        "comments": item.get("descendants") or 0,
    }


def fetch_from_api(item_ids: list[int], *, workers: int = 32, timeout: int = 10) -> pd.DataFrame:
    """Fetch stories directly from the Hacker News API.

    Args:
        item_ids: Item ids to try. Most will be comments and are dropped.
        workers: Concurrent requests. The API is unauthenticated and tolerates this
            comfortably; it is not a rate-limited endpoint.
        timeout: Per-request timeout in seconds.

    Returns:
        A frame in the same shape as the mirror.

    This is what makes "reproducible" mean re-derivable rather than re-downloadable.
    It is bounded and used to check the mirror against its own upstream, because
    fetching 870k items one at a time takes hours and buys nothing a spot check does
    not.
    """
    with requests.Session() as session, ThreadPoolExecutor(max_workers=workers) as pool:
        # requests defaults its pool to 10 connections however many threads use it, so
        # the other 22 workers open a socket, get it discarded, and reconnect for every
        # single request. It still works, which is why it is easy to miss.
        adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
        session.mount("https://", adapter)
        rows = [
            row
            for row in pool.map(lambda i: _fetch_item(i, session, timeout), item_ids)
            if row is not None
        ]
    logger.info("Fetched %d stories from the API out of %d ids", len(rows), len(item_ids))
    return pd.DataFrame(rows, columns=_COLUMNS)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw stories into the shape the feature stage reads.

    Args:
        frame: Raw stories, as published by the mirror or returned by the API.

    Returns:
        A frame with `id`, `title` (normalised, original case kept), `tokens`,
        `url`, `domain`, `score`, `time`, `posted_at`, `author` and `comments`.

    Case is preserved on `title` and lowercasing happens only in `tokens`, because
    capitalisation is a feature: an all-caps title behaves differently from a sentence
    one, and folding case in the ETL would make that unrecoverable downstream.
    """
    frame = frame.loc[:, [column for column in _COLUMNS if column in frame.columns]].copy()
    frame = frame.dropna(subset=["title", "score"])
    frame = frame[frame["title"].astype(str).str.strip() != ""]

    # Deduplicate on id. The mirror is an append-only export, so a story revised after
    # its first capture can appear twice; the later row carries the settled score.
    frame = frame.sort_values("time").drop_duplicates(subset="id", keep="last")

    frame["title"] = frame["title"].astype(str).map(normalise_text)
    frame = frame[frame["title"] != ""]
    frame["tokens"] = frame["title"].map(tokenise)
    frame["url"] = frame["url"].fillna("").astype(str)
    frame["domain"] = frame["url"].map(registered_domain)
    frame["score"] = frame["score"].astype("int64")
    frame["time"] = frame["time"].astype("int64")
    frame["posted_at"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame["author"] = frame["author"].fillna("").astype(str)
    frame["comments"] = frame.get("comments", 0)
    frame["comments"] = frame["comments"].fillna(0).astype("int64")

    # Sorted by (time, id), and the id tie-break is load-bearing rather than tidiness:
    # the author features are an expanding window over this order, so posts sharing a
    # timestamp would otherwise be ordered arbitrarily and the feature would differ
    # between machines. The determinism the whole split rests on starts here.
    frame = frame.sort_values(["time", "id"]).reset_index(drop=True)
    logger.info("Cleaned to %d stories", len(frame))
    return frame


def split_by_time(frame: pd.DataFrame, train_end: str, val_end: str) -> dict[str, pd.DataFrame]:
    """Split chronologically into train, val and test.

    Args:
        frame: A cleaned frame.
        train_end: Posts strictly before this UTC date are training data.
        val_end: Posts before this and on/after `train_end` are validation.

    Returns:
        `{"train": ..., "val": ..., "test": ...}`.

    Chronological rather than random. Hacker News score norms drift over time and posts
    from the same day share a front page, so a random split lets a model see the future
    of its own test set and reports a number that will not survive contact with a real
    post.
    """
    train_boundary = pd.Timestamp(train_end, tz="UTC")
    val_boundary = pd.Timestamp(val_end, tz="UTC")
    return {
        "train": frame[frame["posted_at"] < train_boundary].reset_index(drop=True),
        "val": frame[
            (frame["posted_at"] >= train_boundary) & (frame["posted_at"] < val_boundary)
        ].reset_index(drop=True),
        "test": frame[frame["posted_at"] >= val_boundary].reset_index(drop=True),
    }


def build_splits(config: Config, frame: pd.DataFrame) -> SplitSizes:
    """Write the derived splits and the title corpus to the local cache.

    Args:
        config: Supplies the split dates and the cache directory.
        frame: A cleaned frame.

    Returns:
        The row count of each split.
    """
    config.data_cache_dir.mkdir(parents=True, exist_ok=True)
    splits = split_by_time(frame, config.train_end, config.val_end)
    for name, split in splits.items():
        split.to_parquet(config.split_path(name), index=False)
        logger.info("%-5s %7d rows -> %s", name, len(split), config.split_path(name))

    # Only the training titles. The vocabulary and the vectors are both fitted state:
    # building them over val or test would leak the evaluation into the model.
    with config.titles_path.open("w", encoding="utf-8") as handle:
        for tokens in splits["train"]["tokens"]:
            handle.write(" ".join(tokens) + "\n")
    logger.info("Wrote the training title corpus to %s", config.titles_path)

    return SplitSizes(len(splits["train"]), len(splits["val"]), len(splits["test"]))


def _mirror_upstream(config: Config) -> None:
    """Copy the upstream dataset into our own mirror. Run once, by hand.

    Args:
        config: Supplies `upstream_repo_id`, `hn_repo_id` and the local path.

    Raises:
        FileNotFoundError: If the upstream file cannot be fetched.

    The upstream README says the dataset is updated regularly. Depending on it directly
    would mean the test set quietly changes under a published number, with nothing
    failing to signal it -- which is the same class of problem as the text8 corpus that
    used to be fetched from a teammate's personal repo.
    """
    from huggingface_hub import hf_hub_download

    source = hf_hub_download(config.upstream_repo_id, "story.parquet", repo_type="dataset")
    config.raw_posts_file.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    frame.to_parquet(config.raw_posts_file, index=False)
    logger.info("Copied %d rows from %s", len(frame), config.upstream_repo_id)

    if config.hn_repo_id is None:
        logger.warning("hn_repo_id is unset; the mirror was written locally only")
        return
    pushed = push_checkpoint(
        config.hn_repo_id, config.raw_posts_file, private=False, repo_type="dataset"
    )
    logger.info("Mirror push to %s: %s", config.hn_repo_id, "ok" if pushed else "failed")


def main() -> None:
    """Run the ETL from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("mirror", "api", "upstream"),
        default="mirror",
        help="mirror: the published copy. api: re-derive from Hacker News. "
        "upstream: create the mirror (needs a token; run once).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=2000,
        help="With --source=api, how many ids to fetch.",
    )
    arguments = parser.parse_args()
    config = Config()

    if arguments.source == "upstream":
        _mirror_upstream(config)
        return

    if arguments.source == "api":
        mirror = load_raw(config)
        sample = mirror["id"].tail(arguments.max_items).astype(int).tolist()
        fetched = clean_frame(fetch_from_api(sample))
        expected = clean_frame(mirror[mirror["id"].isin(sample)])
        merged = fetched.merge(expected, on="id", suffixes=("_api", "_mirror"))
        agree = int((merged["title_api"] == merged["title_mirror"]).sum())
        logger.info(
            "API cross-check: %d of %d ids present in both, %d titles identical",
            len(merged),
            len(sample),
            agree,
        )
        return

    sizes = build_splits(config, clean_frame(load_raw(config)))
    logger.info(
        "Splits: train %d, val %d, test %d (total %d)",
        sizes.train,
        sizes.val,
        sizes.test,
        sizes.total,
    )


if __name__ == "__main__":
    main()
