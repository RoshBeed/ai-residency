"""Tests for the frontend's display helpers, project metadata, and backend client.

The frontend had no test suite for one structural reason: `app.py` executes its whole
page flow at import, so nothing defined in it could be imported without a Streamlit
runtime. Splitting the display helpers, the metadata and the HTTP client into their own
Streamlit-free modules is what makes this file possible.
"""

import io
from typing import Any
from unittest.mock import patch

import pytest
import requests
from PIL import Image

import client
import components
from projects import API_URLS, PAGES, PROJECTS, tag_link

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------


def test_every_page_has_a_backend_url() -> None:
    """A page with no API_URLS entry raises KeyError on the readiness check."""
    assert set(PAGES) - {"home"} <= set(API_URLS)


def test_every_backend_has_project_metadata() -> None:
    """PROJECTS drives the home-page cards and each page's sidebar."""
    assert set(API_URLS) == set(PROJECTS)


def test_service_names_match_the_repository_layout() -> None:
    """These keys are also directory names, compose service names and Deployment names.

    They are the join between the frontend, the chart and the repo, so a typo here is
    a 404 in one of three places rather than an error anywhere.
    """
    expected = {
        "word-embeddings",
        "content-scoring",
        "vector-search-retrieval",
        "vit-classifier",
        "vit-seq2seq",
        "multimodal-captioning",
        "whisper-fine-tuning",
        "speech-emotion-recognition",
        "rlhf-ppo",
    }
    assert set(API_URLS) == expected


def test_project_entries_are_complete() -> None:
    for name, project in PROJECTS.items():
        for field in ("icon", "title", "subtitle", "description", "techniques"):
            assert project.get(field), f"{name} is missing {field}"


def test_tag_link_links_known_tags_and_plainly_renders_unknown_ones() -> None:
    assert "<a href=" in tag_link("Embeddings")
    assert tag_link("Not A Real Tag") == '<span class="tag-chip">Not A Real Tag</span>'


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def test_esc_escapes_markup() -> None:
    """Backend output is interpolated into HTML rendered with unsafe_allow_html."""
    assert components.esc("<script>x</script>") == "&lt;script&gt;x&lt;/script&gt;"


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_seq_step_stays_on_the_ramp(probability: float) -> None:
    assert components.seq_step(probability).startswith("#")


def test_seq_step_handles_the_endpoint_without_going_out_of_range() -> None:
    """probability=1.0 indexes past the end unless it is clamped."""
    assert components.seq_step(1.0) == components.seq_step(0.99)


def test_text_color_flips_on_the_dark_steps() -> None:
    light = components.text_color_for(components.seq_step(0.0))
    dark = components.text_color_for(components.seq_step(1.0))
    assert light != dark


def test_render_probability_bars_marks_the_prediction() -> None:
    html = components.render_probability_bars({"happy": 0.7, "sad": 0.3}, "happy")
    assert "happy" in html and "sad" in html
    assert "pred" in html


def test_render_probability_bars_escapes_labels() -> None:
    html = components.render_probability_bars({"<b>x</b>": 1.0}, None)
    assert "<b>x</b>" not in html


def test_render_stats_renders_every_pair() -> None:
    html = components.render_stats([("Accuracy", "91.78%"), ("Baseline", "11.35%")])
    assert "91.78%" in html and "Baseline" in html


def test_render_diff_marks_a_changed_word() -> None:
    html = components.render_diff("My name is Roosh", "My name is Rosh")
    assert "Rosh" in html


def test_render_loss_curve_is_svg() -> None:
    svg = components.render_loss_curve([(0, 2.0), (1, 1.4), (2, 0.9)])
    assert svg.lstrip().startswith("<svg") or "<svg" in svg


def test_render_loss_curve_survives_a_single_point() -> None:
    """One point makes the x-range zero, which divides by zero if unguarded."""
    assert components.render_loss_curve([(0, 1.0)])


def test_draw_patch_grid_renders_at_the_upscaled_size() -> None:
    """Resizes to the model's real input resolution, then overlays its real patch grid."""
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), "white").save(buffer, format="PNG")

    overlay = components.draw_patch_grid(buffer.getvalue(), img_size=28, patch_size=7, upscale=6)
    assert overlay.size == (28 * 6, 28 * 6)


# ---------------------------------------------------------------------------
# Backend client
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_call_returns_the_body_on_success() -> None:
    with patch("client.requests.post", return_value=FakeResponse(200, {"a": 1})):
        result = client.call("word-embeddings", "/nearest", {"word": "python"})
    assert result.ok
    assert result.data == {"a": 1}


def test_call_reports_503_as_not_ready_rather_than_as_an_error() -> None:
    """Every untrained service answers 503, so this is the most common non-200."""
    with patch("client.requests.post", return_value=FakeResponse(503)):
        result = client.call("word-embeddings", "/nearest", {"word": "python"})
    assert result.status == "not_ready"
    assert not result.ok
    assert "trained" in result.message


def test_call_surfaces_the_api_detail_on_a_client_error() -> None:
    response = FakeResponse(404, {"detail": "word 'zzz' not in vocabulary"})
    with patch("client.requests.post", return_value=response):
        result = client.call("word-embeddings", "/nearest", {"word": "zzz"})
    assert result.status == "failed"
    assert "not in vocabulary" in result.message


def test_call_reports_an_unreachable_backend_without_raising() -> None:
    error = requests.exceptions.ConnectionError("refused")
    with patch("client.requests.post", side_effect=error):
        result = client.call("word-embeddings", "/nearest", {"word": "python"})
    assert result.status == "failed"
    assert "word-embeddings" in result.message


def test_call_names_the_timeout_it_gave_up_after() -> None:
    with patch("client.requests.post", side_effect=requests.exceptions.Timeout()):
        result = client.call("rlhf-ppo", "/generate", {}, timeout=42)
    assert result.status == "failed"
    assert "42s" in result.message


def test_call_handles_a_200_that_is_not_json() -> None:
    with patch("client.requests.post", return_value=FakeResponse(200, None, "<html>")):
        result = client.call("word-embeddings", "/nearest", {})
    assert result.status == "failed"
    assert "not JSON" in result.message


def test_is_ready_is_false_for_503() -> None:
    """The readiness contract: 503 means do not route here."""
    with patch("client.requests.get", return_value=FakeResponse(503)):
        assert client.is_ready("word-embeddings") is False


def test_is_ready_is_true_for_200() -> None:
    with patch("client.requests.get", return_value=FakeResponse(200, {})):
        assert client.is_ready("word-embeddings") is True


def test_is_ready_is_false_when_the_backend_is_down() -> None:
    with patch("client.requests.get", side_effect=requests.exceptions.ConnectionError()):
        assert client.is_ready("word-embeddings") is False
