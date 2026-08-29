"""Whisper fine-tuning: transcription and a live training loop."""

from itertools import pairwise
from typing import Any

import requests
import streamlit as st

from client import stream
from components import render_diff, render_loss_curve, render_stats
from shared import bytes_to_b64, esc, service_page, show_stream_error

SERVICE = "whisper-fine-tuning"


def render() -> None:
    with service_page(SERVICE):
        col_record, col_upload = st.columns(2)
        with col_record:
            audio = st.audio_input("Record audio")
            if audio is not None:
                st.audio(audio)
        with col_upload:
            audio_file = st.file_uploader(
                "Upload audio",
                type=["mp3", "wav", "m4a", "aiff", "flac"],
                label_visibility="collapsed",
            )
            if audio_file is not None:
                st.audio(audio_file)

        audio_source = audio or audio_file
        audio_bytes: bytes | None = audio_source.getvalue() if audio_source else None

        target_text = st.text_input(
            "Fine-tune target text",
            value="My name is Rosh",
            help="What the model is fine-tuned to transcribe the clip as. The default is "
            "the demo's own ground truth for the bundled 'My name is Rosh' clip -- if you "
            "record something else, try changing this to match it, or leave it as-is to "
            "see what happens when the target doesn't match what was actually said.",
        )
        if audio_bytes is not None and st.button("Transcribe", type="primary"):
            st.caption("Live -- each event below arrives the moment that stage finishes:")
            tracker = st.empty()
            live_curve = st.empty()
            live_table = st.empty()
            baseline_loss = None
            base_output = None
            transcript_lines: list[str] = []
            steps: list[dict[str, Any]] = []
            try:
                for event in stream(
                    SERVICE,
                    "/transcribe/stream",
                    {"audio_data": bytes_to_b64(audio_bytes), "target_text": target_text},
                    timeout=120,
                ):
                    stage = event["stage"]
                    if stage == "base_transcript":
                        base_output = event["text"]
                        transcript_lines.append(
                            f'<span class="done">✓</span> Base whisper: "<em>{esc(base_output)}</em>"'
                        )
                        transcript_lines.append(
                            '<span class="active">⟳</span> Computing baseline loss…'
                        )
                        tracker.markdown(
                            f'<div class="tracker">{"<br>".join(transcript_lines)}</div>',
                            unsafe_allow_html=True,
                        )
                    elif stage == "baseline_loss":
                        baseline_loss = event["loss"]
                        transcript_lines[-1] = (
                            f'<span class="done">✓</span> Baseline loss: {baseline_loss:.4f}'
                        )
                        transcript_lines.append('<span class="active">⟳</span> Step 1/5…')
                        tracker.markdown(
                            f'<div class="tracker">{"<br>".join(transcript_lines)}</div>',
                            unsafe_allow_html=True,
                        )
                    elif stage == "step":
                        steps.append(event)
                        transcript_lines[-1] = (
                            f'<span class="done">✓</span> Step {event["step"]}/5: '
                            f"loss {event['loss']:.4f}"
                        )
                        if event["step"] < 5:
                            transcript_lines.append(
                                f'<span class="active">⟳</span> Step {event["step"] + 1}/5…'
                            )
                        tracker.markdown(
                            f'<div class="tracker">{"<br>".join(transcript_lines)}</div>',
                            unsafe_allow_html=True,
                        )
                        points = [(0, baseline_loss)] + [(s["step"], s["loss"]) for s in steps]
                        live_curve.markdown(render_loss_curve(points), unsafe_allow_html=True)
                        live_table.dataframe(
                            [{"step": 0, "loss": baseline_loss, "text": base_output}, *steps],
                            use_container_width=True,
                            hide_index=True,
                        )
                    elif stage == "done":
                        st.session_state["audio-fine-tuning_data"] = event
                        tracker.empty()
                        live_curve.empty()
                        live_table.empty()
                    elif stage == "error":
                        tracker.empty()
                        st.error(f"Fine-tuning failed: {event['detail']}")
            except requests.exceptions.RequestException as error:
                show_stream_error(error, SERVICE)

        data = st.session_state.get("audio-fine-tuning_data")
        if data:
            st.markdown(
                '<div class="trace-caption">'
                "Not a cached demo. Fine-tunes Whisper-tiny from scratch each request — "
                f'5 gradient steps toward target "{esc(target_text)}".</div>',
                unsafe_allow_html=True,
            )
            points = [(0, data["baseline_loss"])] + [
                (s["step"], s["loss"]) for s in data["fine_tune_trace"]
            ]
            st.markdown(render_loss_curve(points), unsafe_allow_html=True)
            st.markdown(
                render_stats(
                    [
                        ("Baseline loss (step 0)", f"{data['baseline_loss']:.4f}"),
                        (
                            "Final loss",
                            f"{data['fine_tune_trace'][-1]['loss']:.4f}"
                            if data["fine_tune_trace"]
                            else "n/a",
                        ),
                    ]
                ),
                unsafe_allow_html=True,
            )

            step_rows = [
                {"step": 0, "loss": data["baseline_loss"], "text": data["base_output"]}
            ] + [
                {"step": s["step"], "loss": s["loss"], "text": s["text"]}
                for s in data["fine_tune_trace"]
            ]
            st.caption("Per-step transcript — green = added, red = dropped:")
            for prev_row, curr_row in pairwise(step_rows):
                loss = curr_row["loss"]
                st.markdown(f"**Step {curr_row['step']}** (loss = {loss:.4f})")
                st.markdown(render_diff(prev_row["text"], curr_row["text"]), unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Base Whisper**")
                st.text(data["base_output"])
            with col2:
                st.markdown("**Fine-tuned (5 steps)**")
                st.text(data["ft_output"])
            with st.expander("Raw response"):
                st.json(data)
