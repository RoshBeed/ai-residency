"""Vision Transformer encoder-decoder: image to digit sequence."""

from typing import Any

import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from client import stream
from components import draw_patch_grid, render_digit_trace
from shared import bytes_to_b64, canvas_to_png_bytes, service_page

SERVICE = "vit-seq2seq"


def render() -> None:
    with service_page(SERVICE):
        col_draw, col_upload = st.columns(2)
        with col_draw:
            st.caption("Draw a multi-digit number")
            seq_canvas = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=14,
                stroke_color="#ffffff",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="vit_seq2seq_canvas",
            )
        with col_upload:
            st.caption("Or upload one")
            seq_uploaded = st.file_uploader(
                "Upload a multi-digit image (PNG/JPG)",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )
            if seq_uploaded is not None:
                st.image(seq_uploaded, width=150)

        seq_bytes: bytes | None = None
        if seq_canvas.image_data is not None:
            seq_bytes = canvas_to_png_bytes(seq_canvas.image_data)
        elif seq_uploaded is not None:
            seq_bytes = seq_uploaded.getvalue()

        if seq_bytes is not None and st.button("Read the number", type="primary"):
            b64_seq = bytes_to_b64(seq_bytes)
            st.caption("Input: 84x84px, 16x16 patches. Decoded autoregressively.")
            st.image(draw_patch_grid(seq_bytes, 84, 16), width=260)

            st.caption("Live -- each digit below appears the moment the model emits it:")
            digit_display = st.empty()
            seq2seq_data = None
            digit_trace_sofar: list[dict[str, Any]] = []
            try:
                for event in stream(
                    SERVICE,
                    "/predict/stream",
                    {"image_data": b64_seq},
                    timeout=30,
                ):
                    stage = event["stage"]
                    if stage == "digit" and event["digit"]:
                        digit_trace_sofar.append(
                            {"digit": event["digit"], "confidence": event["confidence"]}
                        )
                        digit_display.markdown(
                            render_digit_trace(digit_trace_sofar),
                            unsafe_allow_html=True,
                        )
                    elif stage == "done":
                        seq2seq_data = event
                        digit_display.empty()
                    elif stage == "error":
                        digit_display.empty()
                        st.error(f"Seq2seq generation failed: {event['detail']}")
            except requests.exceptions.RequestException as e:
                st.error(f"SSE request failed: {e}")

            if seq2seq_data:
                st.markdown(f"**Predicted: {seq2seq_data['pred']}**")
                if seq2seq_data.get("digit_trace"):
                    st.caption("Darker = higher confidence at each step:")
                    st.markdown(
                        render_digit_trace(seq2seq_data["digit_trace"]), unsafe_allow_html=True
                    )
                with st.expander("Raw SSE response"):
                    st.json(seq2seq_data)
