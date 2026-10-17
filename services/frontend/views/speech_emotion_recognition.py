"""Speech emotion recognition over Whisper features."""

import streamlit as st

from client import call
from components import render_probability_bars
from shared import bytes_to_b64, service_page, show

SERVICE = "speech-emotion-recognition"


def render() -> None:
    with service_page(SERVICE):
        st.caption(
            "Record a clip or upload one. The model predicts which of six emotions it hears, "
            "by scoring the emotion tokens its decoder can emit."
        )
        # The browser records at its own rate, usually 48kHz, and the clip goes to the API
        # as-is. `speech.load_audio` resamples to the model's 16kHz on the way in, so this
        # needs no conversion here -- and must not do one, or there would be two places
        # that decide the sample rate.
        recorded = st.audio_input("Record a clip")
        emotion_upload = st.file_uploader("Or upload audio (WAV/MP3)", type=["wav", "mp3", "m4a"])

        # A recording wins when both are present: it is the thing just produced, and the
        # uploader keeps its file across reruns.
        clip = recorded or emotion_upload
        if clip is not None:
            st.audio(clip)

        if clip is not None and st.button("Predict emotion", type="primary"):
            # 120s: this one loads Whisper and runs it per request.
            result = show(
                call(
                    SERVICE,
                    "/predict",
                    {"audio_data": bytes_to_b64(clip.getvalue())},
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
