"""Speech emotion recognition over Whisper features."""

import streamlit as st

from client import call
from components import render_probability_bars
from shared import bytes_to_b64, service_page, show

SERVICE = "speech-emotion-recognition"


def render() -> None:
    with service_page(SERVICE):
        st.caption(
            "Upload a clip. The model predicts which of six emotions it hears, by scoring the "
            "emotion tokens its decoder can emit."
        )
        emotion_upload = st.file_uploader(
            "Upload audio (WAV/MP3)", type=["wav", "mp3", "m4a"], label_visibility="collapsed"
        )
        if emotion_upload is not None:
            st.audio(emotion_upload)

        if emotion_upload is not None and st.button("Predict emotion", type="primary"):
            # 120s: this one loads Whisper and runs it per request.
            result = show(
                call(
                    SERVICE,
                    "/predict",
                    {"audio_data": bytes_to_b64(emotion_upload.getvalue())},
                    timeout=120,
                )
            )
            if result:
                st.markdown(f"**Predicted: {result['emotion']}**")
                st.caption(f"Checkpoint source: `{result['checkpoint_source']}`")
                st.markdown(
                    render_probability_bars(result["scores"], result["emotion"]),
                    unsafe_allow_html=True,
                )
                with st.expander("Raw response"):
                    st.json(result)
