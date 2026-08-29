"""RLHF with PPO: generation, reward, KL and the policy update."""

import math
from typing import Any

import requests
import streamlit as st

from client import stream
from components import render_diverging_token_heatmap, seq_step, text_color_for
from shared import esc, service_page, show_stream_error

SERVICE = "rlhf-ppo"

# The twelve stages a PPO step streams, in order, with the field each one adds to
# an (s, a) pair and the expression it implements. Module level: they describe the
# backend's contract, not anything this render pass computes.

_PPO_STAGE_LABELS = {
    "generate": "1. Sampling",
    "kl_divergence": "2. Token-Level KL Penalty",
    "rewards": "3. Reward Scoring",
    "adjusted_reward": "4. KL-Penalized Reward",
    "value": "5. Value Function",
    "td_error": "6. Temporal Difference Error",
    "advantage": "7. Generalized Advantage Estimation",
    "reward_to_go": "8. Rewards-to-Go",
    "value_loss": "9. Value Function Loss",
    "clip": "10. Clip",
    "policy_loss": "11. PPO-Clip Loss",
    "policy_ratio": "12. Policy Ratio",
}


_PPO_STAGE_FIELD = {
    "kl_divergence": "kl_div",
    "adjusted_reward": "adjusted_reward",
    "value": "value",
    "td_error": "td_error",
    "advantage": "advantage",
    "reward_to_go": "reward_to_go",
    "clip": "clipped_ratio",
    "policy_ratio": "policy_ratio",
}


_RENDERING = {
    "kl_divergence": ("2. Token-Level KL Penalty", "kl_div"),
    "adjusted_reward": ("4. KL-Penalized Reward", "adjusted_reward"),
    "value": ("5. Value Function", "value"),
    "td_error": ("6. Temporal Difference Error", "td_error"),
    "advantage": ("7. Generalized Advantage Estimation", "advantage"),
    "reward_to_go": ("8. Rewards-to-Go", "reward_to_go"),
    "clip": ("10. Clip", "clipped_ratio"),
    "policy_ratio": ("12. Policy Ratio", "policy_ratio"),
}


_MATH = {
    "generate": "a_t  ∼  π_θ(·∣s_t)",
    "kl_divergence": "log π_θ(a_t∣s_t) / π_ref(a_t∣s_t)",
    "rewards": "r_ψ(x, y)",
    "adjusted_reward": "r − η·KL_t",
    "value": "V_ϕ(s_t)",
    "td_error": "δ_t  =  r_t + γV_ϕ(s_{t+1}) − V_ϕ(s_t)",
    "advantage": "Â_t  =  Σ_{l≥0} (γλ)^l δ_{t+l}",
    "reward_to_go": "R̂_t  =  Â_t + V_ϕ(s_t)",
    "value_loss": "L_V  =  E[(V_ϕ − R̂_t)²]    ···  ∂V",
    "clip": "clamp(r_t(θ), 1±ε)",
    "policy_loss": "L_CLIP  =  E[min(r_tÂ_t, clip(r_t, 1±ε)Â_t)]    ···  ∂π",
    "policy_ratio": "r_t(θ)  =  π_θ / π_old    ——  readout",
}


def render() -> None:
    with service_page(SERVICE):
        ppo_steps = st.number_input("PPO steps", min_value=1, max_value=20, value=5, step=1)
        if st.button("Run PPO steps", type="primary"):
            st.caption("Live — 12 stages per step. Token rows persist across stages.")

            try:
                status_line = st.empty()
                unified = st.empty()
                blocks: list[str] = []
                last_load: str = ""
                sampling_chips: list[str] = []

                # Per-step metrics for the post-run summary card.
                step_records: list[dict[str, float | str]] = []
                cur: dict[str, float | str] = {}

                def _render_heatmap(
                    label: str, stage: str, field: str, event: dict[str, Any], math: dict[str, Any]
                ) -> str:
                    html = (
                        f'<div class="stage-block">'
                        f'<span class="stage-label">{label}</span>'
                        f'<span class="stage-math">{math[stage]}</span>'
                        f"</div>"
                    )
                    pairs_by_prompt: dict[int, list[dict[str, Any]]] = {}
                    for pair in event["sa_pairs"]:
                        pairs_by_prompt.setdefault(pair["prompt_idx"], []).append(pair)
                    for pidx in sorted(pairs_by_prompt):
                        ordered = sorted(pairs_by_prompt[pidx], key=lambda p: p["step"])
                        html += render_diverging_token_heatmap(
                            [p["action_token"] for p in ordered],
                            [p[field] for p in ordered],
                        )
                    return html

                def flush() -> None:
                    unified.markdown("".join(blocks), unsafe_allow_html=True)

                for event in stream(
                    SERVICE,
                    "/ppo_step/stream",
                    {
                        "prompts": [
                            "SUBREDDIT: r/relationships\nTITLE: Should I admit to snooping?\nPOST: ...\nTL;DR:"
                        ],
                        "ppo_steps": ppo_steps,
                    },
                    timeout=600,
                ):
                    stage = event["stage"]
                    if stage == "loading":
                        last_load = event["detail"]
                        status_line.markdown(f"⏳ {last_load}")
                    elif stage == "step_boundary":
                        status_line.empty()
                        if cur:
                            step_records.append(cur)
                            cur = {}
                        step_num = event["step"]
                        step_total = event["total"]
                        blocks.append(
                            f'<div class="step-header">PPO Step {step_num} / {step_total}</div>'
                        )
                        flush()
                    elif stage == "generating":
                        prob = math.exp(event["logprob"])
                        bg = seq_step(prob)
                        fg = text_color_for(bg)
                        token = esc(event["token"]) or "&nbsp;"
                        sampling_chips.append(
                            f'<span class="token-chip" style="background:{bg};color:{fg}" '
                            f'title="{prob * 100:.1f}% confidence">{token}</span>'
                        )
                        sampling_block = (
                            f'<div class="stage-block">'
                            f'<span class="stage-label">1. Sampling</span>'
                            f'<span class="stage-math">{_MATH["generate"]}</span>'
                            f"</div>"
                            f'<div class="token-row">{"".join(sampling_chips)}</div>'
                        )
                        if len(blocks) == 0 or "1. Sampling" in blocks[0]:
                            blocks[:1] = [sampling_block]
                        else:
                            blocks.insert(0, sampling_block)
                        status_line.empty()
                        flush()
                    elif stage == "generate" and "sa_pairs" in event:
                        status_line.empty()
                        pairs = sorted(event["sa_pairs"], key=lambda p: p["step"])
                        chips: list[str] = []
                        for pair in pairs:
                            prob = math.exp(pair["logprob"])
                            bg = seq_step(prob)
                            fg = text_color_for(bg)
                            token = esc(pair["action_token"]) or "&nbsp;"
                            chips.append(
                                f'<span class="token-chip" style="background:{bg};color:{fg}" '
                                f'title="{prob * 100:.1f}% confidence">{token}</span>'
                            )
                        completion = esc(event["generated_texts"][0])
                        cur["completion"] = completion
                        sampling_block = (
                            f'<div class="stage-block">'
                            f'<span class="stage-label">1. Sampling</span>'
                            f'<span class="stage-math">{_MATH["generate"]}</span>'
                            f'<span class="stage-completion">{completion}</span>'
                            f"</div>"
                            f'<div class="token-row">{"".join(chips)}</div>'
                        )
                        if len(blocks) > 0 and "1. Sampling" in blocks[0]:
                            blocks[0] = sampling_block
                        else:
                            blocks.append(sampling_block)
                        sampling_chips = []
                        flush()
                    elif stage == "heartbeat":
                        if last_load:
                            status_line.markdown(f"⏳ {last_load}  —  {event['elapsed']}s")
                    elif stage == "done":
                        if cur:
                            step_records.append(cur)
                        st.session_state["rlhf-ppo_ppo_data"] = event
                        st.session_state["rlhf-ppo_step_records"] = step_records
                        status_line.empty()
                    elif stage == "error":
                        status_line.empty()
                        st.error(f"PPO step failed: {event['detail']}")
                    elif stage == "rewards" and "rewards" in event:
                        cur["reward"] = event["rewards"][0]
                        vals = ", ".join(f"{r:.3f}" for r in event["rewards"])
                        blocks.append(
                            f'<div class="stage-block">'
                            f'<span class="stage-label">3. Reward Scoring</span>'
                            f'<span class="stage-math">{_MATH["rewards"]}</span>'
                            f"</div>"
                            f'<div class="stage-scalar">{vals}</div>'
                        )
                        flush()
                    elif stage in {"value_loss", "policy_loss"}:
                        label = (
                            "9. Value Function Loss"
                            if stage == "value_loss"
                            else "11. PPO-Clip Loss"
                        )
                        math_key = "value_loss" if stage == "value_loss" else "policy_loss"
                        value = (
                            event.get("value_loss")
                            if stage == "value_loss"
                            else event.get("policy_loss")
                        )
                        if value is not None:
                            if stage == "value_loss":
                                cur["value_loss"] = value
                            else:
                                cur["policy_loss"] = value
                            blocks.append(
                                f'<div class="stage-block">'
                                f'<span class="stage-label">{label}</span>'
                                f'<span class="stage-math">{_MATH[math_key]}</span>'
                                f"</div>"
                                f'<div class="stage-scalar">{value:.4f}</div>'
                            )
                            flush()
                    elif stage == "kl_divergence" and "sa_pairs" in event:
                        kls = [p["kl_div"] for p in event["sa_pairs"]]
                        cur["avg_kl"] = sum(kls) / len(kls) if kls else 0.0
                        label, field = _RENDERING[stage]
                        block_html = _render_heatmap(label, stage, field, event, _MATH)
                        blocks.append(block_html)
                        flush()
                    elif stage == "policy_ratio" and "sa_pairs" in event:
                        ratios = [p["policy_ratio"] for p in event["sa_pairs"]]
                        cur["avg_ratio"] = sum(ratios) / len(ratios) if ratios else 1.0
                        label, field = _RENDERING[stage]
                        block_html = _render_heatmap(label, stage, field, event, _MATH)
                        blocks.append(block_html)
                        # Per-step summary line.
                        step_reward = cur.get("reward", float("nan"))
                        vl = cur.get("value_loss", float("nan"))
                        pl = cur.get("policy_loss", float("nan"))
                        summary = '<div class="step-summary">'
                        summary += "<table><tr>"
                        summary += f"<td>reward</td><td>{step_reward:.2f}</td>"
                        summary += f"<td>value loss</td><td>{vl:.1f}</td>"
                        summary += f"<td>policy loss</td><td>{pl:+.3f}</td>"
                        summary += f"<td>avg KL</td><td>{cur['avg_kl']:.2f}</td>"
                        summary += f"<td>avg ratio</td><td>{cur['avg_ratio']:.3f}</td>"
                        summary += "</tr></table></div>"
                        blocks.append(summary)
                        flush()
                    elif stage in _RENDERING and "sa_pairs" in event:
                        label, field = _RENDERING[stage]
                        block_html = _render_heatmap(label, stage, field, event, _MATH)
                        blocks.append(block_html)
                        flush()
            except requests.exceptions.RequestException as error:
                show_stream_error(error, SERVICE)

        ppo_data = st.session_state.get("rlhf-ppo_ppo_data")
        if ppo_data:
            records = st.session_state.get("rlhf-ppo_step_records", [])
            if records and len(records) > 1:
                r_first = records[0].get("reward", float("nan"))
                r_last = records[-1].get("reward", float("nan"))
                vl_first = records[0].get("value_loss", float("nan"))
                vl_last = records[-1].get("value_loss", float("nan"))
                pl_first = records[0].get("policy_loss", float("nan"))
                pl_last = records[-1].get("policy_loss", float("nan"))
                kl_first = records[0].get("avg_kl", float("nan"))
                kl_last = records[-1].get("avg_kl", float("nan"))

                def _arrow(a: float, b: float) -> str:
                    if math.isnan(a) or math.isnan(b) or abs(b - a) < 0.01:
                        return " →"
                    return " ↑" if b > a else " ↓"

                st.markdown(
                    f'<div class="summary-card">'
                    f"<h4>{len(records)} PPO steps — before → after</h4>"
                    f'<table class="summary-table">'
                    f'<tr><th style="text-align:left;font-weight:500;color:var(--text-muted)">metric</th>'
                    f'<th style="text-align:right;font-weight:500;color:var(--text-muted)">step 1</th>'
                    f'<th style="text-align:right;font-weight:500;color:var(--text-muted)">step {len(records)}</th></tr>'
                    f"<tr><td>Reward</td>"
                    f'<td style="text-align:right">{r_first:.2f}</td>'
                    f'<td style="text-align:right">{r_last:.2f}'
                    f'<span class="{"summary-good" if r_last > r_first else "summary-bad" if r_last < r_first else ""}">{_arrow(r_first, r_last)}</span></td></tr>'
                    f"<tr><td>Value loss</td>"
                    f'<td style="text-align:right">{vl_first:.1f}</td>'
                    f'<td style="text-align:right">{vl_last:.1f}'
                    f'<span class="summary-{"good" if vl_last < vl_first else "bad" if vl_last > vl_first else ""}">{_arrow(vl_first, vl_last)}</span></td></tr>'
                    f"<tr><td>Policy loss</td>"
                    f'<td style="text-align:right">{pl_first:+.3f}</td>'
                    f'<td style="text-align:right">{pl_last:+.3f}</td></tr>'
                    f"<tr><td>KL divergence</td>"
                    f'<td style="text-align:right">{kl_first:.2f}</td>'
                    f'<td style="text-align:right">{kl_last:.2f}'
                    f'<span class="">{_arrow(kl_first, kl_last)}</span></td></tr>'
                    f"</table></div>",
                    unsafe_allow_html=True,
                )
            st.info(ppo_data["note"])
            with st.expander("Raw response"):
                st.json(ppo_data)
