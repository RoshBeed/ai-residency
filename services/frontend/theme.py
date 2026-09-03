"""The logo and the shared component CSS.

Separated from `app.py` for the same reason `projects.py` is: it is a large block of
static content with no logic, and keeping it here leaves the page flow readable.
"""

import base64

__all__ = ["LOGO_URI", "PALETTE_CSS"]

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
  <rect width="96" height="96" rx="18" fill="#2a78d6"/>
  <text x="48" y="38" text-anchor="middle" fill="white" font-size="22" font-weight="600" font-family="system-ui,-apple-system,sans-serif">ML</text>
  <text x="48" y="70" text-anchor="middle" fill="#b7d3f6" font-size="14" font-family="system-ui,-apple-system,sans-serif">folio</text>
</svg>"""
LOGO_URI = f"data:image/svg+xml;base64,{base64.b64encode(LOGO_SVG.encode()).decode()}"

# ---------------------------------------------------------------------------
# Palette + shared component CSS, injected once. Sequential magnitude (word/
# token/digit confidence, patch-grid lines) stays one hue (blue) light->dark;
# known/unknown is a status pair (green/red), always paired with an icon so
# it's never color-alone. Both light and dark surfaces are declared, keyed
# off prefers-color-scheme, so this reads correctly in either OS theme.
# ---------------------------------------------------------------------------
PALETTE_CSS = """
<style>
/* Reduce default top padding so cards sit above the fold */
.main .block-container { padding-top: 1rem !important; }
h1 { text-align: center; }

/* Equal-height cards: force Streamlit columns to stretch */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    display: flex;
    flex-direction: column;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div:first-child {
    flex: 1;
}

:root {
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --gridline: #e1e0d9;
  --series-1: #2a78d6;
  --series-1-muted: #b7d3f6;
  --good: #0ca30c;
  --critical: #d03b3b;
  --chip-bg: #f9f9f7;
  --border: rgba(11,11,11,0.10);
  --tag-technique-bg: #d4ddf7;
  --tag-technique-fg: #2a4cc7;
  --tag-model-bg: #f5e6d3;
  --tag-model-fg: #b86e1c;
  --tag-paper-bg: #e8e0f0;
  --tag-paper-fg: #6b4c8a;
  --tag-data-bg: #d4ece2;
  --tag-data-fg: #2d7a5f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --series-1: #3987e5;
    --series-1-muted: #184f95;
    --good: #0ca30c;
    --critical: #e66767;
    --chip-bg: #202020;
    --border: rgba(255,255,255,0.10);
    --tag-technique-bg: #1f387a;
    --tag-technique-fg: #8dbcf7;
    --tag-model-bg: #4d3514;
    --tag-model-fg: #f0b84d;
    --tag-paper-bg: #3a284d;
    --tag-paper-fg: #c4a3de;
    --tag-data-bg: #1f4733;
    --tag-data-fg: #6fd69a;
  }
}

/* ---- Home page: hero, cards, tags ---- */

.hero-section { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero-title { font-size: 2.6rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.4rem; letter-spacing: -0.02em; }
.hero-tagline { font-size: 0.95rem; color: var(--text-secondary); max-width: 720px; line-height: 1.55; margin: 0 auto 1.5rem; }

.card { border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px 10px; background: var(--chip-bg); width: 100%; min-width: 0; box-sizing: border-box; }
.card-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.card-icon { font-size: 1.3rem; flex-shrink: 0; }
.card-title { font-size: 0.95rem; font-weight: 600; color: var(--text-primary); }
.card-desc { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.45; margin-bottom: 10px; }
.tag-row { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 6px; }
.tag-chip { font-size: 11px; padding: 3px 10px; border-radius: 12px; border: 1px solid transparent; background: var(--tag-technique-bg); color: var(--tag-technique-fg); font-weight: 500; white-space: nowrap; }
.tag-link { text-decoration: none; }
.tag-link:hover { filter: brightness(0.9); }
.tag-model { font-size: 11px; padding: 3px 10px; border-radius: 12px; border: 1px solid transparent; background: var(--tag-model-bg); color: var(--tag-model-fg); font-weight: 500; white-space: nowrap; text-decoration: none; }
.tag-model:hover { filter: brightness(0.9); }

.tag-paper { font-size: 11px; padding: 3px 10px; border-radius: 12px; border: 1px solid transparent; background: var(--tag-paper-bg); color: var(--tag-paper-fg); font-weight: 500; white-space: nowrap; text-decoration: none; }
.tag-paper:hover { filter: brightness(0.9); }
.tag-data { font-size: 11px; padding: 3px 10px; border-radius: 12px; border: 1px solid transparent; background: var(--tag-data-bg); color: var(--tag-data-fg); font-weight: 500; white-space: nowrap; text-decoration: none; }
.tag-data:hover { filter: brightness(0.9); }
.card-actions { text-align: right; margin-top: 2px; }
.card-link { font-size: 11px; color: var(--text-muted); text-decoration: none; font-weight: 500; }
.card-link:hover { color: var(--series-1); }

/* ---- Demo-page shared components ---- */

.trace-caption { font-size: 13px; color: var(--text-muted); margin: 4px 0 10px; }
.trace-row, .token-row { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 14px; }
.trace-chip {
  font-size:13px; padding:4px 10px; border-radius:6px; background:var(--chip-bg);
  border:1px solid var(--border); color:var(--text-secondary);
}
.trace-chip.known { border-left:3px solid var(--good); }
.trace-chip.unknown { border-left:3px solid var(--critical); color:var(--text-muted); text-decoration:line-through; }
.token-chip {
  font-size:14px; padding:4px 8px; border-radius:5px; font-variant-numeric:tabular-nums;
  cursor:default; border:1px solid var(--border);
}
.digit-chip {
  font-size:22px; font-weight:600; padding:10px 16px; border-radius:6px;
  cursor:default; border:1px solid var(--border);
}
.prob-chart { margin:6px 0 14px; }
.prob-row { display:flex; align-items:center; gap:8px; margin:3px 0; }
.prob-label {
  min-width:60px; max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  font-size:13px; color:var(--text-secondary); font-variant-numeric:tabular-nums;
}
.prob-track { flex:1; height:14px; background:var(--gridline); border-radius:4px; overflow:hidden; }
.prob-fill { height:100%; background:var(--series-1-muted); }
.prob-fill.pred { background:var(--series-1); }
.prob-value { width:56px; text-align:right; font-size:12px; color:var(--text-muted); font-variant-numeric:tabular-nums; }
.prob-delta { width:46px; text-align:right; font-size:11px; color:var(--text-muted); font-variant-numeric:tabular-nums; }
.loss-svg { display:block; margin:6px 0 4px; }
.stat-row { display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px; }
.stat-tile { padding:10px 16px; border:1px solid var(--border); border-radius:8px; background:var(--chip-bg); }
.stat-tile .label { font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:.03em; }
.stat-tile .value { font-size:20px; color:var(--text-primary); font-variant-numeric:tabular-nums; }
.diff-box {
  padding:10px 14px; border:1px solid var(--border); border-radius:8px;
  background:var(--chip-bg); margin:4px 0 14px; line-height:1.7; font-size:14px;
  color:var(--text-secondary);
}
.diff-ins { color:var(--good); text-decoration:underline; font-weight:600; }
.diff-del { color:var(--critical); text-decoration:line-through; opacity:0.7; }

/* ---- Live SSE progress tracker + score bars ---- */

.tracker { font-family: ui-monospace,SFMono-Regular,monospace; font-size: 13px; line-height: 1.85; margin: 4px 0 14px; color: var(--text-secondary); }
.tracker .done { color: var(--good); }
.tracker .active { color: var(--series-1); font-weight: 500; }
.tracker .pending { color: var(--text-muted); }

.score-bars { margin: 6px 0 14px; }
.score-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.score-rank { min-width: 36px; font-size: 12px; color: var(--text-muted); text-align: right; font-variant-numeric: tabular-nums; }
.score-fill { height: 14px; border-radius: 3px; min-width: 4px; transition: width 0.4s ease; }
.score-fill.best { background: #0ca30c; }
.score-fill.mid { background: var(--series-1-muted); }
.score-fill.worst { background: var(--gridline); }
.score-doc { font-size: 13px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.score-val { font-size: 11px; color: var(--text-muted); font-variant-numeric: tabular-nums; width: 50px; text-align: right; }
.score-detail { font-size: 12px; color: var(--text-muted); line-height: 1.5; margin: 2px 0 8px 44px; }

.sim-matrix { border-collapse: collapse; margin: 10px 0 16px; font-size: 12px; font-variant-numeric: tabular-nums; }
.sim-matrix th { font-weight: 500; color: var(--text-muted); padding: 4px 8px; text-align: center; }
.sim-matrix td { padding: 6px 10px; text-align: center; border-radius: 2px; cursor: default; }

.stage-block { margin: 6px 0 14px; }
.stage-label { font-weight: 600; font-size: 14px; }
.stage-detail { font-size: 12px; color: var(--text-muted); }
.stage-math { font-size: 12px; color: var(--text-muted); }
.stage-completion { font-size: 13px; color: var(--text-secondary); margin-left: 4px; font-style: italic; }
.stage-scalar { font-size: 15px; color: var(--series-1); font-variant-numeric: tabular-nums; margin-left: 4px; }
.status-line { font-size: 14px; color: var(--text-muted); margin: 4px 0; }

.step-header {
  background: var(--series-1-muted); color: var(--series-1);
  padding: 8px 12px; border-radius: 4px; margin: 18px 0 8px;
  font-weight: 600; font-size: 14px;
}
.step-summary { font-size: 12px; color: var(--text-secondary); margin: 4px 0 14px 12px; }
.step-summary td { padding: 1px 8px; font-variant-numeric: tabular-nums; }
.step-summary .trend-up { color: var(--good); }
.step-summary .trend-down { color: var(--critical); }
.step-summary .trend-flat { color: var(--text-muted); }

.arch-table { border-collapse: collapse; margin: 4px 0 10px; font-size: 12px; }
.arch-table td, .arch-table th { padding: 4px 10px; text-align: left; border-bottom: 1px solid var(--border); }
.arch-table th { color: var(--text-muted); font-weight: 500; }

.summary-card {
  border: 1px solid var(--border); border-radius: 6px; padding: 14px 18px;
  margin: 14px 0; background: var(--bg-light, #fafaf9);
}
.summary-card h4 { margin: 0 0 8px; font-size: 14px; }
.summary-table { border-collapse: collapse; font-size: 13px; width: 100%; }
.summary-table td { padding: 3px 10px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
.summary-table td:first-child { color: var(--text-muted); }
.summary-table .summary-good { color: var(--good); }
.summary-table .summary-bad { color: var(--critical); }

/* Sidebar tertiary buttons: strip chrome so they read as plain text links */
section[data-testid="stSidebar"] .stButton button[kind="tertiary"] {
    background: none; border: none; padding: 3px 0; margin: 0;
    font-size: 0.95rem; text-align: left; line-height: 1.5;
    color: var(--text-secondary); min-height: unset;
}
section[data-testid="stSidebar"] .stButton button[kind="tertiary"]:hover {
    color: var(--series-1); text-decoration: underline; background: none;
}
</style>
"""
