"""The corpus pipeline: fetch Hacker News and text8, clean and lemmatize the text,
and merge it into one corpus plus word/lemma vocabularies.

``TextProcessor`` normalizes, cleans and lemmatizes text. ``DatabaseProcessor``
talks to the read-only Hacker News mirror. ``HNPipeline`` and ``WikiPipeline``
turn those two sources into word/lemma vocabularies, and ``build_corpus`` merges
them into the single corpus the CBOW stage trains on. This file is deliberately
the unglamorous one: it is all plumbing, no model logic.
"""

import html
import json
import os
import re
import string
import threading
import time
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import nltk
import psycopg2
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from psycopg2.extras import RealDictCursor
from tqdm import tqdm

from ai_residency_common import get_logger, resolve_checkpoint
from config import Config

logger = get_logger(__name__)

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)  # NLTK >=3.9 renamed the resource
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)


class TextProcessor:
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.lock = threading.Lock()
        self.pos_tag_lock = threading.Lock()

        # Define POS tag mapping
        self.POS_TAG_MAP = {
            "J": "a",  # adjective
            "N": "n",  # noun
            "V": "v",  # verb
            "R": "r",  # adverb
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text by converting special characters to their closest ASCII equivalent."""

        # First, normalize Unicode characters
        text = unicodedata.normalize("NFKD", text)

        # Convert to ASCII, ignoring non-ASCII characters
        # The ASCII strip now happens AFTER the replacement table below.
        # Running it here deleted every character that table matches, so the
        # whole table was dead code: 'cost: EUR20 - GBP15' came out 'cost: 20  15'.

        # Replace common special characters with their ASCII equivalents
        replacements = {
            '"': '"',
            """: "'",
            """: "'",
            "–": "-",
            "—": "-",
            "…": "...",
            "→": "->",
            "←": "<-",
            "↑": "^",
            "↓": "v",
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
            "²": "2",
            "³": "3",
            "µ": "u",
            "¶": "P",
            "·": ".",
            "¹": "1",
            "¼": "1/4",
            "½": "1/2",
            "¾": "3/4",
            "≠": "!=",
            "≤": "<=",
            "≥": ">=",
            "≈": "~=",
            "∞": "inf",
            "∑": "sum",
            "∏": "prod",
            "√": "sqrt",
            "∫": "integral",
            "∆": "delta",
            "∇": "nabla",
            "∂": "partial",
            "∝": "propto",
            "∅": "empty",
            "∈": "in",
            "∉": "not in",
            "∋": "contains",
            "∌": "not contains",
            "⊂": "subset",
            "⊃": "superset",
            "⊆": "subset or equal",
            "⊇": "superset or equal",
            "⊕": "xor",
            "⊗": "tensor",
            "⊥": "perp",
            "⊤": "top",
            "⊢": "proves",
            "⊨": "models",
            "⊩": "forces",
            "⊪": "double turnstile",
            "⊫": "double turnstile",
            "⊬": "does not prove",
            "⊭": "does not model",
            "⊮": "does not force",
            "⊯": "does not double turnstile",
            "⊰": "precedes under relation",
            "⊱": "succeeds under relation",
            "⊲": "normal subgroup of",
            "⊳": "contains as normal subgroup",
            "⊴": "normal subgroup of or equal to",
            "⊵": "contains as normal subgroup or equal to",
            "⊶": "original of",
            "⊷": "image of",
            "⊸": "multimap",
            "⊹": "hermitian conjugate matrix",
            "⊺": "intercalate",
            "⊻": "xor",
            "⊼": "nand",
            "⊽": "nor",
            "⊾": "right angle with arc",
            "⊿": "right triangle",
            "⋀": "n-ary logical and",
            "⋁": "n-ary logical or",
            "⋂": "n-ary intersection",
            "⋃": "n-ary union",
            "⋄": "diamond operator",
            "⋅": "dot operator",
            "⋆": "star operator",
            "⋇": "division times",
            "⋈": "bowtie",
            "⋉": "left normal factor semidirect product",
            "⋊": "right normal factor semidirect product",
            "⋋": "left semidirect product",
            "⋌": "right semidirect product",
            "⋍": "reversed tilde equals",
            "⋎": "curly logical or",
            "⋏": "curly logical and",
            "⋐": "double subset",
            "⋑": "double superset",
            "⋒": "double intersection",
            "⋓": "double union",
            "⋔": "pitchfork",
            "⋕": "equal and parallel to",
            "⋖": "less than with dot",
            "⋗": "greater than with dot",
            "⋘": "very much less than",
            "⋙": "very much greater than",
            "⋚": "less than equal to or greater than",
            "⋛": "greater than equal to or less than",
            "⋜": "equal to or less than",
            "⋝": "equal to or greater than",
            "⋞": "equal to or precedes",
            "⋟": "equal to or succeeds",
            "⋠": "does not precede or equal",
            "⋡": "does not succeed or equal",
            "⋢": "not square image of or equal to",
            "⋣": "not square original of or equal to",
            "⋤": "square image of or not equal to",
            "⋥": "square original of or not equal to",
            "⋦": "less than but not equivalent to",
            "⋧": "greater than but not equivalent to",
            "⋨": "precedes but not equivalent to",
            "⋩": "succeeds but not equivalent to",
            "⋪": "not normal subgroup of",
            "⋫": "does not contain as normal subgroup",
            "⋬": "not normal subgroup of or equal to",
            "⋭": "does not contain as normal subgroup or equal",
            "⋮": "vertical ellipsis",
            "⋯": "midline horizontal ellipsis",
            "⋰": "up right diagonal ellipsis",
            "⋱": "down right diagonal ellipsis",
            "⋲": "element of with long horizontal stroke",
            "⋳": "element of with vertical bar at end of horizontal stroke",
            "⋴": "small element of with vertical bar at end of horizontal stroke",
            "⋵": "element of with dot above",
            "⋶": "element of with overbar",
            "⋷": "small element of with overbar",
            "⋸": "element of with underbar",
            "⋹": "element of with two horizontal strokes",
            "⋺": "contains with long horizontal stroke",
            "⋻": "contains with vertical bar at end of horizontal stroke",
            "⋼": "small contains with vertical bar at end of horizontal stroke",
            "⋽": "contains with overbar",
            "⋾": "small contains with overbar",
            "⋿": "z notation bag membership",
        }

        for special, replacement in replacements.items():
            text = text.replace(special, replacement)

        # Anything still non-ASCII had no mapping, so drop it.
        text = text.encode("ascii", "ignore").decode("ascii")

        return text

    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning:
        1. Convert HTML entities to Unicode
        2. Convert Unicode escape sequences
        3. Remove HTML tags and their content
        4. Convert to lowercase
        5. Remove URLs
        6. Remove punctuation and special characters
        7. Remove numbers
        8. Remove extra whitespace
        """
        if not text:
            return ""

        # First normalize the text to handle special characters
        text = self.normalize_text(text)

        # Convert HTML entities to Unicode
        text = html.unescape(text)

        # Convert Unicode escape sequences
        try:
            text = text.encode("utf-8").decode("unicode-escape")
        except UnicodeError:
            # If decoding fails, try to clean the text directly
            text = text.encode("utf-8", errors="ignore").decode("utf-8")

        # Remove HTML tags and their content (more aggressive)
        # First remove complete tags with content
        text = re.sub(r"<[^>]*>.*?</[^>]*>", " ", text)  # Add space after removing tags
        # Then remove any remaining tags
        text = re.sub(r"<[^>]+>", " ", text)  # Add space after removing tags
        # Remove any remaining HTML entities
        text = re.sub(r"&[a-zA-Z]+;", " ", text)  # Add space after removing entities
        text = re.sub(r"&#[0-9]+;", " ", text)  # Add space after removing entities
        text = re.sub(r"&#x[0-9a-fA-F]+;", " ", text)  # Add space after removing entities

        # Remove URLs (more aggressive)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # Add space after removing URLs

        # Remove any remaining HTML-like patterns
        text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)  # Add space after removing entities
        text = re.sub(r"<[^>]*>", " ", text)  # Add space after removing tags
        # Space, not empty string, so removing a URL does not fuse its neighbours.
        text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', " ", text)

        # Remove any remaining HTML-like patterns
        text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)  # Add space after removing entities
        text = re.sub(r"<[^>]*>", " ", text)  # Add space after removing tags
        # Space, not empty string, so removing a URL does not fuse its neighbours.
        text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', " ", text)

        # Remove any remaining HTML-like patterns
        text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)  # Add space after removing entities
        text = re.sub(r"<[^>]*>", " ", text)  # Add space after removing tags
        # Space, not empty string, so removing a URL does not fuse its neighbours.
        text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', " ", text)

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters, but preserve spaces
        text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

        # Remove numbers
        text = re.sub(r"\d+", " ", text)  # Add space after removing numbers

        # Remove extra whitespace and normalize spaces
        text = " ".join(text.split())

        # Final normalization to catch any remaining special characters
        text = self.normalize_text(text)

        # Ensure proper spacing between words
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # Add space between camelCase
        text = re.sub(r"([a-z])(\d)", r"\1 \2", text)  # Add space between letters and numbers
        text = re.sub(r"(\d)([a-z])", r"\1 \2", text)  # Add space between numbers and letters

        # Final cleanup of any remaining multiple spaces
        text = " ".join(text.split())

        return text.strip()

    def get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to first character lemmatize() accepts."""
        with self.pos_tag_lock:
            tag = nltk.pos_tag([word])[0][1][0].upper()
        return self.POS_TAG_MAP.get(tag, "n")  # Default to noun if tag not found

    def lemmatize_word(self, word: str) -> str:
        """Lemmatize a single word using its POS tag."""
        cleaned_word = self.clean_text(word)
        if not cleaned_word:
            return word

        try:
            pos = self.get_wordnet_pos(cleaned_word)
            lemma: str = self.lemmatizer.lemmatize(cleaned_word, pos)
            return lemma
        except Exception:
            return cleaned_word  # Return cleaned word if lemmatization fails

    def process_batch(self, words: list[str]) -> dict[str, str]:
        """Process a batch of words and return word to lemma mapping."""
        return {word: self.lemmatize_word(word) for word in words}

    def lemmatize_word_index_dict(
        self, word_to_index: dict[str, int], num_threads: int = 4
    ) -> dict[str, int]:
        """Lemmatize words in parallel and create word to lemma index mapping."""
        logger.info("Lemmatizing words...")
        words = list(word_to_index.keys())
        batch_size = len(words) // num_threads + 1
        word_to_lemma_index = {}

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(0, len(words), batch_size):
                batch = words[i : i + batch_size]
                futures.append(executor.submit(self.process_batch, batch))

            for future in tqdm(as_completed(futures), total=len(futures)):
                batch_results = future.result()
                with self.lock:
                    word_to_lemma_index.update(batch_results)

        # Create lemma to index mapping
        lemma_to_index: dict[str, int] = {}
        for lemma in word_to_lemma_index.values():
            if lemma not in lemma_to_index:
                lemma_to_index[lemma] = len(lemma_to_index)

        # Update word_to_lemma_index to use lemma indices
        return {word: lemma_to_index[lemma] for word, lemma in word_to_lemma_index.items()}

    def process_chunk(self, chunk: str) -> tuple[int, Counter[str]]:
        """Process a chunk of text and return word count and frequency."""
        cleaned_text = self.clean_text(chunk)
        words = cleaned_text.split()
        return len(words), Counter(words)


# The read-only Hacker News mirror the course provided. Hardcoded here originally,
# which made the corpus stage unrunnable anywhere the host is unreachable -- and it is
# unreachable from at least one network this repo has been developed on. Override with
# HN_DATABASE_URL to point at your own copy.
DEFAULT_CONNECTION_STRING = "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"

# How long to wait before deciding the mirror is not there. Without this psycopg2
# inherits the OS default and a blocked port hangs the stage for minutes rather than
# failing in a way the caller can act on.
CONNECT_TIMEOUT_SECONDS = 10


class DatabaseProcessor:
    def __init__(self, connection_string: str | None = None) -> None:
        self.connection_string = (
            connection_string or os.environ.get("HN_DATABASE_URL") or DEFAULT_CONNECTION_STRING
        )
        self.text_processor = TextProcessor()

    def get_connection(self) -> Any:
        """Create and return a database connection."""
        return psycopg2.connect(self.connection_string, connect_timeout=CONNECT_TIMEOUT_SECONDS)

    def is_reachable(self) -> bool:
        """Whether the mirror can actually be connected to.

        Returns:
            True if a connection opens within the timeout, False otherwise.
        """
        try:
            self.get_connection().close()
        except psycopg2.Error as error:
            logger.warning("Hacker News mirror unreachable: %s", str(error).strip())
            return False
        return True

    def execute_query(
        self, query: str, params: tuple[Any, ...] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as a list of dictionaries."""
        with (
            self.get_connection() as conn,
            conn.cursor(cursor_factory=RealDictCursor) as cur,
        ):
            cur.execute(query, params)
            rows: list[dict[str, Any]] = cur.fetchall()
            return rows

    def get_hn_data(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get Hacker News data from multiple tables."""
        all_results = []

        # List of tables to query
        tables = [
            "hacker_news.items_by_month_2024_10",
            "hacker_news.items_by_month_2024_09",
            "hacker_news.items_by_month_2024_08",
            "hacker_news.items_by_month_2024_07",
            "hacker_news.items_by_month_2024_06",
            "hacker_news.items_by_month_2024_05",
            "hacker_news.items_by_month_2024_04",
            "hacker_news.items_by_month_2024_03",
            "hacker_news.items_by_month_2024_02",
            "hacker_news.items_by_month_2024_01",
        ]

        # Calculate limit per table
        per_table_limit = limit // len(tables) if limit else None

        for table in tables:
            query = f"""
                SELECT 
                    id, type, by, title, text, parent, kids, 
                    url, score, descendants, time
                FROM {table}
            """
            if per_table_limit:
                query += f" LIMIT {per_table_limit}"

            results = self.execute_query(query)
            all_results.extend(results)
            logger.info(f"Fetched {len(results)} rows from {table}")

        return all_results

    def save_raw_data_to_json(self, data: list[dict[str, Any]], output_path: str) -> None:
        """Save raw data to JSON file."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert datetime objects to ISO format strings and normalize text
        def json_handler(obj: Any) -> str:
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, str):
                return self.text_processor.normalize_text(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Save raw data to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=json_handler)

    def process_hn_data(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process raw Hacker News data from database."""
        processed_posts = []

        # First, identify all stories (posts)
        stories = [row for row in results if row["type"] == "story" and row["title"] is not None]

        for story in stories:
            # For stories, use URL if available, otherwise use text (for Ask HN posts)
            post_content = ""
            if story.get("url"):
                post_content = story["url"]
            elif story.get("text"):
                post_content = story["text"]

            post = {
                "Title": self.clean_text(story["title"]),
                "Author": story["by"],
                "Content": self.clean_text(post_content),
                "Score": story["score"],
                "Comments": [],
            }

            # Find all comments for this story
            def get_comments(parent_id: Any) -> list[str]:
                comments: list[str] = []
                for row in results:
                    if (
                        row["type"] == "comment"
                        and row["parent"] == parent_id
                        and row["text"] is not None
                    ):
                        comments.append(self.clean_text(row["text"]))
                        # Recursively get nested comments
                        if row["kids"]:
                            comments.extend(get_comments(row["id"]))
                return comments

            # Get all comments for this story
            post["Comments"] = get_comments(story["id"])
            processed_posts.append(post)

        return processed_posts

    def create_massive_string(self, data: list[dict[str, Any]]) -> str:
        """
        Creates a massive string from the cleaned data by concatenating all text content.
        Each post and its comments are separated by spaces.
        Excludes author names and URLs.
        """
        massive_string = []

        for post in data:
            # Add title only (no author)
            massive_string.append(post["Title"])

            # Add content only if it's not a URL
            if post["Content"] and not post["Content"].startswith("http"):
                massive_string.append(post["Content"])

            # Add comments
            if post["Comments"]:
                for comment in post["Comments"]:
                    massive_string.append(comment)

        return " ".join(massive_string)

    def save_hn_data_to_file(self, output_path: str, limit: int | None = None) -> None:
        """Save Hacker News data to a text file."""
        # Get raw data
        results = self.get_hn_data(limit)

        # Save raw data to JSON first
        raw_json_path = os.path.join(os.path.dirname(output_path), "hn_data_raw.json")
        self.save_raw_data_to_json(results, raw_json_path)
        logger.info(f"Raw data saved to: {raw_json_path}")

        # Process and clean the raw data
        processed_posts = self.process_hn_data(results)

        # Create and save massive string
        massive_string = self.create_massive_string(processed_posts)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(massive_string)

    def clean_text(self, text: str) -> str:
        """Clean text using the TextProcessor."""
        return self.text_processor.clean_text(text)


class BasePipeline(ABC):
    def __init__(self, output_dir: str = "data", chunk_size: int = 1024 * 1024) -> None:
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.word_count = 0
        self.word_freq: Counter[str] = Counter()
        self.lock = threading.Lock()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def get_data(self) -> str:
        """Get data from source (file path or database)"""
        pass

    def process_chunk(self, chunk: str) -> tuple[int, Counter[str]]:
        """Process a chunk of text and return word count and frequency."""
        words = chunk.split()
        return len(words), Counter(words)

    def process_file_chunk(self, start_pos: int, end_pos: int, file_path: str) -> None:
        """Process a specific portion of the file."""
        with open(file_path, encoding="utf-8") as file:
            file.seek(start_pos)
            chunk = file.read(end_pos - start_pos)
            word_count, word_freq = self.process_chunk(chunk)

            with self.lock:
                self.word_count += word_count
                self.word_freq.update(word_freq)

    def tokenize(self, file_path: str, num_threads: int = 4) -> tuple[int, dict[str, int]]:
        """Tokenize the file using multiple threads."""
        logger.info("Tokenizing text...")
        start_time = time.time()

        file_size = os.path.getsize(file_path)
        chunk_size = file_size // num_threads

        threads = []
        for i in range(num_threads):
            start_pos = i * chunk_size
            end_pos = start_pos + chunk_size if i < num_threads - 1 else file_size

            thread = threading.Thread(
                target=self.process_file_chunk, args=(start_pos, end_pos, file_path)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()
        logger.info(f"Tokenization completed in {end_time - start_time:.2f} seconds")

        return self.word_count, dict(self.word_freq)

    def save_results(
        self, word_to_index: dict[str, int], word_to_lemma_index: dict[str, int], prefix: str
    ) -> None:
        """Save processing results to files."""
        # Save word to index mapping
        index_path = os.path.join(self.output_dir, f"{prefix}_word_to_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(word_to_index, f, indent=2, ensure_ascii=False)

        # Save lemma index mapping
        lemma_path = os.path.join(self.output_dir, f"{prefix}_word_to_lemma_index.json")
        with open(lemma_path, "w", encoding="utf-8") as f:
            json.dump(word_to_lemma_index, f, indent=2, ensure_ascii=False)

    @abstractmethod
    def run(self, num_threads: int = 4) -> tuple[dict[str, int], dict[str, int]]:
        """Run the pipeline and return word_to_index and word_to_lemma_index mappings."""
        pass


class HNPipeline(BasePipeline):
    """Builds the Hacker News half of the corpus."""

    def __init__(self, output_dir: str = "data", limit: int | None = None) -> None:
        super().__init__(output_dir)
        self.text_processor = TextProcessor()
        self.db_processor = DatabaseProcessor()
        self.hn_data_path = os.path.join(output_dir, "hn_data.txt")
        self.limit = limit

    def get_data(self) -> str:
        """Get HN data, from the mirror if it is reachable and from disk if not.

        The mirror is a course-provided host that is not reachable from every network.
        When it is not, an earlier cleaned extract on disk is used instead, so the
        corpus stage completes rather than taking the whole pipeline down with it.

        Returns:
            The corpus text built from the posts.

        Raises:
            FileNotFoundError: If the mirror is unreachable and no extract exists.
        """
        if not self.db_processor.is_reachable():
            return self._data_from_existing_extract()

        logger.info("Fetching HN data from multiple tables...")
        # Get raw data
        results = self.db_processor.get_hn_data(limit=self.limit)
        logger.info(f"Total rows fetched: {len(results)}")

        # Save raw data to JSON first
        raw_json_path = os.path.join(os.path.dirname(self.hn_data_path), "hn_data_raw.json")
        self.db_processor.save_raw_data_to_json(results, raw_json_path)
        logger.info(f"Raw data saved to: {raw_json_path}")

        # Process the raw data
        processed_posts = self.db_processor.process_hn_data(results)
        logger.info("Processed %d posts", len(processed_posts))

        return self._write_corpus(processed_posts)

    def _data_from_existing_extract(self) -> str:
        """Rebuild the corpus from the last cleaned extract on disk.

        Returns:
            Path to the corpus file.

        Raises:
            FileNotFoundError: If there is no extract to fall back to.
        """
        config = Config()
        resolved, source = resolve_checkpoint(
            config.posts_file, config.hf_dataset_repo_id, repo_type="dataset"
        )
        if resolved is None:
            raise FileNotFoundError(
                f"The Hacker News mirror is unreachable, and the extract is neither at "
                f"{config.posts_file} nor in {config.hf_dataset_repo_id}. Set HN_DATABASE_URL "
                f"to a reachable copy of the mirror."
            )
        json_path = str(resolved)
        logger.info("Using the cleaned extract from %s", source)

        with open(json_path, encoding="utf-8") as handle:
            processed_posts: list[dict[str, Any]] = json.load(handle)

        logger.info("Reusing %d posts from %s", len(processed_posts), json_path)
        return self._write_corpus(processed_posts, write_json=False)

    def _write_corpus(
        self, processed_posts: list[dict[str, Any]], *, write_json: bool = True
    ) -> str:
        """Flatten posts into one text file for the CBOW stage.

        Args:
            processed_posts: Cleaned posts, each with Title, Content and Comments.
            write_json: Whether to also persist the cleaned extract. False when the
                posts were just read back out of it.

        Returns:
            Path to the corpus file.
        """
        if write_json:
            json_path = os.path.join(os.path.dirname(self.hn_data_path), "hn_data_cleaned.json")
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(processed_posts, handle, indent=2, ensure_ascii=False)
            logger.info("Wrote %d posts with scores to %s", len(processed_posts), json_path)

        corpus: list[str] = []
        for post in processed_posts:
            corpus.append(post["Title"])
            # Skip bare URLs: a link is not language, and they were a real share of
            # the token count.
            if post["Content"] and not post["Content"].startswith("http"):
                corpus.append(post["Content"])
            corpus.extend(post["Comments"] or [])

        os.makedirs(os.path.dirname(self.hn_data_path), exist_ok=True)
        with open(self.hn_data_path, "w", encoding="utf-8") as handle:
            handle.write(" ".join(corpus))

        logger.info("Corpus written to %s", self.hn_data_path)
        return self.hn_data_path

    def process_chunk(self, chunk: str) -> tuple[int, Counter[str]]:
        """Process a chunk of text with cleaning."""
        return self.text_processor.process_chunk(chunk)

    def run(self, num_threads: int = 4) -> tuple[dict[str, int], dict[str, int]]:
        """Run the HN pipeline."""
        # Get data
        file_path = self.get_data()

        # Tokenize
        word_count, word_freq = self.tokenize(file_path, num_threads)
        logger.info(f"Total words: {word_count}")
        logger.info(f"Unique words: {len(word_freq)}")

        # Create word to index mapping
        word_to_index = {word: idx for idx, word in enumerate(word_freq.keys())}

        # Lemmatize
        word_to_lemma_index = self.text_processor.lemmatize_word_index_dict(
            word_to_index, num_threads
        )

        # Save results
        self.save_results(word_to_index, word_to_lemma_index, "hn")

        return word_to_index, word_to_lemma_index


class WikiPipeline(BasePipeline):
    def __init__(self, output_dir: str = "data") -> None:
        super().__init__(output_dir)
        self.text_processor = TextProcessor()
        self.text8_url = (
            "https://huggingface.co/datasets/roshbeed/ai-residency-text8/resolve/main/text8"
        )
        self.text8_path = os.path.join(output_dir, "text8")

    def get_data(self) -> str:
        """Download text8 dataset if not exists."""
        if not os.path.exists(self.text8_path):
            logger.info("Downloading text8 dataset...")
            response = requests.get(self.text8_url, stream=True)
            response.raise_for_status()

            with open(self.text8_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Download completed.")

        return self.text8_path

    def run(self, num_threads: int = 4) -> tuple[dict[str, int], dict[str, int]]:
        """Run the Wiki pipeline."""
        # Get data
        file_path = self.get_data()

        # Tokenize
        word_count, word_freq = self.tokenize(file_path, num_threads)
        logger.info(f"Total words: {word_count}")
        logger.info(f"Unique words: {len(word_freq)}")

        # Create word to index mapping
        word_to_index = {word: idx for idx, word in enumerate(word_freq.keys())}

        # Lemmatize
        word_to_lemma_index = self.text_processor.lemmatize_word_index_dict(
            word_to_index, num_threads
        )

        # Save results
        self.save_results(word_to_index, word_to_lemma_index, "wiki")

        return word_to_index, word_to_lemma_index


def merge_vocabularies(
    wiki_word_to_index: dict[str, int],
    wiki_word_to_lemma_index: dict[str, int],
    hn_word_to_index: dict[str, int],
    hn_word_to_lemma_index: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """Merge vocabularies from both sources."""
    # Create combined word to index mapping
    combined_word_to_index = {}
    current_idx = 0

    # Add Wiki words
    for word in wiki_word_to_index:
        if word not in combined_word_to_index:
            combined_word_to_index[word] = current_idx
            current_idx += 1

    # Add HN words
    for word in hn_word_to_index:
        if word not in combined_word_to_index:
            combined_word_to_index[word] = current_idx
            current_idx += 1

    # Create combined lemma to index mapping
    lemma_to_index = {}
    current_lemma_idx = 0

    # Process all words
    for word in combined_word_to_index:
        # Try to get lemma from either source
        lemma = wiki_word_to_lemma_index.get(word) or hn_word_to_lemma_index.get(word)
        if lemma is not None and lemma not in lemma_to_index:
            lemma_to_index[lemma] = current_lemma_idx
            current_lemma_idx += 1

    # Create final word to lemma index mapping
    combined_word_to_lemma_index = {}
    for word in combined_word_to_index:
        lemma = wiki_word_to_lemma_index.get(word) or hn_word_to_lemma_index.get(word)
        if lemma is not None:
            combined_word_to_lemma_index[word] = lemma_to_index[lemma]

    return combined_word_to_index, combined_word_to_lemma_index


def merge_text_data(output_dir: str, combined_word_to_index: dict[str, int]) -> None:
    """Merge text data from both sources into a combined file."""
    logger.info("\nMerging text data...")

    # Read text8 data
    with open(os.path.join(output_dir, "text8"), encoding="utf-8") as f:
        wiki_text = f.read()

    # Read HN data
    with open(os.path.join(output_dir, "hn_data.txt"), encoding="utf-8") as f:
        hn_text = f.read()

    # Combine texts with a space
    combined_text = f"{wiki_text} {hn_text}"

    # Save combined text
    with open(os.path.join(output_dir, "combined_data.txt"), "w", encoding="utf-8") as f:
        f.write(combined_text)

    # Count words in each source
    wiki_words = len(wiki_text.split())
    hn_words = len(hn_text.split())
    total_words = wiki_words + hn_words

    logger.info("Text data merged successfully!")
    logger.info(f"Wiki words: {wiki_words:,}")
    logger.info(f"HN words: {hn_words:,}")
    logger.info(f"Total words: {total_words:,}")


def build_corpus() -> None:
    # Create output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Run Wiki pipeline
    logger.info("Running Wiki pipeline...")
    wiki_pipeline = WikiPipeline(output_dir)
    wiki_word_to_index, wiki_word_to_lemma_index = wiki_pipeline.run(num_threads=4)

    # Run HN pipeline with 10000 row limit
    logger.info("\nRunning HN pipeline (100000 row limit)...")
    hn_pipeline = HNPipeline(output_dir, limit=100000)
    hn_word_to_index, hn_word_to_lemma_index = hn_pipeline.run(num_threads=4)

    # Merge vocabularies
    logger.info("\nMerging vocabularies...")
    combined_word_to_index, combined_word_to_lemma_index = merge_vocabularies(
        wiki_word_to_index, wiki_word_to_lemma_index, hn_word_to_index, hn_word_to_lemma_index
    )

    # Save combined results
    logger.info("Saving combined results...")
    with open(os.path.join(output_dir, "combined_word_to_index.json"), "w", encoding="utf-8") as f:
        json.dump(combined_word_to_index, f, indent=2, ensure_ascii=False)

    with open(
        os.path.join(output_dir, "combined_word_to_lemma_index.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(combined_word_to_lemma_index, f, indent=2, ensure_ascii=False)

    # Merge text data
    merge_text_data(output_dir, combined_word_to_index)

    logger.info("\nProcessing complete!")
    logger.info(f"Total unique words: {len(combined_word_to_index)}")
    logger.info(f"Total unique lemmas: {len(set(combined_word_to_lemma_index.values()))}")


def main() -> None:
    """Entry point for `make train-corpus`."""
    build_corpus()


if __name__ == "__main__":
    main()
