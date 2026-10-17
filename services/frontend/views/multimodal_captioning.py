"""Multimodal captioning: vision encoder into a language model."""

from typing import Any

import requests
import streamlit as st

from client import stream
from components import render_token_heatmap
from shared import esc, image_to_b64, service_page, show_stream_error

SERVICE = "multimodal-captioning"


def render() -> None:
    with service_page(SERVICE):
        col_upload, col_camera = st.columns(2)
        with col_upload:
            uploaded = st.file_uploader(
                "Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
            )
            if uploaded is not None:
                st.image(uploaded, width=250)
        with col_camera:
            camera = st.camera_input("Take a photo", label_visibility="collapsed")
            if camera is not None:
                st.image(camera, width=250)

        image_source = uploaded or camera
        if image_source is not None and st.button("Generate caption", type="primary"):
            st.caption("Live -- each token below appears the moment the model emits it:")
            tracker = st.empty()
            live_heatmap = st.empty()
            tokens: list[dict[str, Any]] = []
            data = None
            try:
                for event in stream(
                    SERVICE,
                    "/caption/stream",
                    # No checkpoint_name: the service picks its own default. Naming one here
                    # pinned the UI to the 10-pair adapter and no retrained checkpoint could
                    # reach it.
                    {"image_data": image_to_b64(image_source)},
                    timeout=90,
                ):
                    stage = event["stage"]
                    if stage == "model_loaded":
                        tracker.markdown(
                            '<div class="tracker">'
                            '<span class="done">✓</span> CLIP + Qwen3 loaded<br>'
                            '<span class="active">⟳</span> Generating caption…'
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    elif stage == "token":
                        tokens.append(
                            {"token": event["token"], "probability": event["probability"]}
                        )
                        live_heatmap.markdown(render_token_heatmap(tokens), unsafe_allow_html=True)
                        tracker.markdown(
                            f'<div class="tracker">'
                            f'<span class="done">✓</span> Models loaded<br>'
                            f'<span class="active">⟳</span> Token {event["step"] + 1}: '
                            f'"<strong>{esc(event["token"])}</strong>" '
                            f"({event['probability'] * 100:.1f}% confidence)"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    elif stage == "done":
                        data = event
                        tracker.markdown(
                            f'<div class="tracker">'
                            f'<span class="done">✓</span> Models loaded<br>'
                            f'<span class="done">✓</span> {len(data["token_trace"])} tokens generated'
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    elif stage == "error":
                        tracker.empty()
                        st.error(f"Captioning failed: {event['detail']}")
            except requests.exceptions.RequestException as error:
                show_stream_error(error, SERVICE)

            if data:
                st.caption(
                    "CLIP image embedding → prepended to prompt → Qwen3 autoregressive decode. "
                    "Darker = higher sampling probability."
                )
                live_heatmap.empty()
                st.markdown(render_token_heatmap(data["token_trace"]), unsafe_allow_html=True)
                st.success(data["caption"])
                st.caption(f"Checkpoint: {data['checkpoint_used']}")
                with st.expander("Raw SSE response"):
                    st.json(data)
