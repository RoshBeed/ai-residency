"""Display helpers: pure functions from backend data to HTML, SVG or an image.

None of these touches Streamlit, the network, or module-level state -- each takes the
data a backend already returned and formats it. That is deliberate, and it is what
makes them testable: `app.py` runs its whole page flow at import, so anything defined
there can only be exercised by starting a Streamlit runtime. Everything here can be
called from a plain unit test.

The colour choices are shared across every page so the same encoding means the same
thing everywhere: one blue ramp for sequential magnitude (confidence, probability), a
green/red pair for the known/unknown status, always paired with an icon so the meaning
never rests on colour alone.
"""

import difflib
import html
import io
from typing import Any

from PIL import Image, ImageDraw

__all__ = [
    "draw_patch_grid",
    "esc",
    "render_diff",
    "render_digit_trace",
    "render_diverging_token_heatmap",
    "render_loss_curve",
    "render_probability_bars",
    "render_stats",
    "render_token_heatmap",
    "render_word_trace",
    "seq_step",
    "text_color_for",
]

# Sequential blue ramp, steps 100->650 from the validated default palette
# (references/palette.md). Index 0 = near-zero, last = near-certain.
_SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
_SEQ_BLUE_DARK_STEPS = {"#3987e5", "#2a78d6", "#1c5cab", "#104281"}
_SEQ_BLUE_RGB = (42, 120, 214)  # #2a78d6, for the PIL patch-grid overlay


def esc(value: object) -> str:
    return html.escape(str(value))


def seq_step(probability: float) -> str:
    """Map a probability in [0, 1] to one step of the sequential blue ramp."""
    index = min(max(int(probability * len(_SEQ_BLUE)), 0), len(_SEQ_BLUE) - 1)
    return _SEQ_BLUE[index]


def text_color_for(hex_bg: str) -> str:
    """White text on the ramp's dark steps, dark text on its light steps."""
    return "#ffffff" if hex_bg in _SEQ_BLUE_DARK_STEPS else "#0b0b0b"


def render_stats(items: list[tuple[str, str]]) -> str:
    tiles = "".join(
        f'<div class="stat-tile"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div></div>'
        for label, value in items
    )
    return f'<div class="stat-row">{tiles}</div>'


def render_word_trace(word_trace: list[dict[str, Any]]) -> str:
    """Known/unknown vocabulary chips -- a status pair, always icon + label."""
    chips = []
    for w in word_trace:
        known = bool(w["known"])
        cls = "known" if known else "unknown"
        icon = "✓" if known else "✗"
        chips.append(f'<span class="trace-chip {cls}">{icon} {esc(w["word"])}</span>')
    return f'<div class="trace-row">{"".join(chips)}</div>'


def render_probability_bars(probs: dict[str, float], predicted: str | None) -> str:
    """Single-series magnitude bars, one hue, the predicted/chosen label emphasized."""
    rows = []
    for label, p in probs.items():
        pred_cls = " pred" if label == predicted else ""
        rows.append(
            f'<div class="prob-row">'
            f'<span class="prob-label" title="{esc(label)}">{esc(label)}</span>'
            f'<div class="prob-track"><div class="prob-fill{pred_cls}" '
            f'style="width:{max(p, 0.0) * 100:.1f}%"></div></div>'
            f'<span class="prob-value">{p * 100:.1f}%</span></div>'
        )
    return f'<div class="prob-chart">{"".join(rows)}</div>'


def render_token_heatmap(tokens: list[dict[str, Any]]) -> str:
    """Each token shaded by its own sampling/greedy probability -- one hue, magnitude only."""
    spans = []
    for t in tokens:
        prob = float(t["probability"])
        bg = seq_step(prob)
        fg = text_color_for(bg)
        label = esc(t["token"]) or "&nbsp;"
        spans.append(
            f'<span class="token-chip" style="background:{bg};color:{fg}" '
            f'title="{prob * 100:.1f}% confidence">{label}</span>'
        )
    return f'<div class="token-row">{"".join(spans)}</div>'


def render_digit_trace(digit_trace: list[dict[str, Any]]) -> str:
    chips = []
    for d in digit_trace:
        conf = float(d["confidence"])
        bg = seq_step(conf)
        fg = text_color_for(bg)
        chips.append(
            f'<span class="digit-chip" style="background:{bg};color:{fg}" '
            f'title="{conf * 100:.1f}% confidence">{esc(d["digit"])}</span>'
        )
    return f'<div class="token-row">{"".join(chips)}</div>'


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


def _lerp_hex(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> str:
    r = round(c1[0] + (c2[0] - c1[0]) * t)
    g = round(c1[1] + (c2[1] - c1[1]) * t)
    b = round(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# Diverging pair (blue <-> red, neutral gray midpoint) for signed per-token
# quantities -- KL divergence, advantages, TD errors, etc. can be negative, so the
# sequential single-hue ramp (magnitude only) doesn't fit; sign matters here too.
_DIVERGING_NEUTRAL = _hex_to_rgb("#f0efec")
_DIVERGING_BLUE = _hex_to_rgb("#2a78d6")
_DIVERGING_RED = _hex_to_rgb("#e34948")


def render_diverging_token_heatmap(tokens: list[str], values: list[float]) -> str:
    """Shade each token by a signed value: red for positive, blue for negative,
    gray at zero, scaled by the largest magnitude in this sequence."""
    max_abs = max((abs(v) for v in values), default=0.0)
    spans = []
    for token, value in zip(tokens, values, strict=True):
        t = 0.0 if max_abs <= 0 else max(-1.0, min(1.0, value / max_abs))
        bg = _lerp_hex(_DIVERGING_NEUTRAL, _DIVERGING_RED if t >= 0 else _DIVERGING_BLUE, abs(t))
        fg = "#ffffff" if abs(t) > 0.55 else "#0b0b0b"
        label = esc(token) or "&nbsp;"
        spans.append(
            f'<span class="token-chip" style="background:{bg};color:{fg}" '
            f'title="{value:+.4f}">{label}</span>'
        )
    return f'<div class="token-row">{"".join(spans)}</div>'


def render_diff(before: str, after: str) -> str:
    """Word-level diff between two transcripts: insertions green+underlined, deletions
    red+struck-through -- both color and a second visual cue, never color alone."""
    before_words = before.split()
    after_words = after.split()
    matcher = difflib.SequenceMatcher(a=before_words, b=after_words)
    spans = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            spans.append(esc(" ".join(before_words[i1:i2])))
        elif tag == "delete":
            spans.append(f'<span class="diff-del">{esc(" ".join(before_words[i1:i2]))}</span>')
        elif tag == "insert":
            spans.append(f'<span class="diff-ins">{esc(" ".join(after_words[j1:j2]))}</span>')
        elif tag == "replace":
            spans.append(f'<span class="diff-del">{esc(" ".join(before_words[i1:i2]))}</span>')
            spans.append(f'<span class="diff-ins">{esc(" ".join(after_words[j1:j2]))}</span>')
    return f'<div class="diff-box">{" ".join(spans) or "&nbsp;"}</div>'


def render_loss_curve(points: list[tuple[int, float]]) -> str:
    """A single-series inline SVG line -- one axis, one hue, hover titles as the tooltip layer."""
    width, height, pad = 460, 170, 32
    values = [v for _, v in points]
    max_v = max(values) if values else 1.0
    max_v = max_v if max_v > 0 else 1.0
    n = len(points)

    def x(i: int) -> float:
        return pad + (width - 2 * pad) * (i / (n - 1) if n > 1 else 0.0)

    def y(v: float) -> float:
        return height - pad - (height - 2 * pad) * (v / max_v)

    coords = [(x(i), y(v)) for i, (_, v) in enumerate(points)]
    poly = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in coords)
    circles = "".join(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="var(--series-1)">'
        f"<title>step {step}: loss {v:.4f}</title></circle>"
        for (step, v), (cx, cy) in zip(points, coords, strict=True)
    )
    labels = "".join(
        f'<text x="{cx:.1f}" y="{height - pad + 16}" font-size="11" fill="var(--text-muted)" '
        f'text-anchor="middle">{step}</text>'
        for (step, _v), (cx, _cy) in zip(points, coords, strict=True)
    )
    return (
        f'<svg width="{width}" height="{height}" class="loss-svg" role="img" '
        f'aria-label="Loss per fine-tuning step">'
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
        f'stroke="var(--gridline)" stroke-width="1"/>'
        f'<polyline points="{poly}" fill="none" stroke="var(--series-1)" stroke-width="2"/>'
        f"{circles}{labels}</svg>"
    )


def draw_patch_grid(
    image_bytes: bytes, img_size: int, patch_size: int, upscale: int = 6
) -> Image.Image:
    """Resize to the model's real input resolution, then overlay its real patch grid."""
    base = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((img_size, img_size))
    overlay = base.resize((img_size * upscale, img_size * upscale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(overlay, "RGBA")
    step = patch_size * upscale
    w, h = overlay.size
    line_color = (*_SEQ_BLUE_RGB, 200)
    for gx in range(0, w + 1, step):
        draw.line([(gx, 0), (gx, h)], fill=line_color, width=1)
    for gy in range(0, h + 1, step):
        draw.line([(0, gy), (w, gy)], fill=line_color, width=1)
    return overlay
