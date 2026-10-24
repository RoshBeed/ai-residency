"""Speech emotion recognition over Whisper features."""

import contextlib
import io
import wave

import streamlit as st

from client import call
from components import render_probability_bars
from shared import bytes_to_b64, service_page, show

SERVICE = "speech-emotion-recognition"


def _describe(clip_bytes: bytes) -> str:
    """Say how long the clip is and at what rate, from the bytes about to be sent.

    The widgets above show whatever the browser produced. This reads the payload the API
    will actually receive, so a recording that lost its opening or a section in the
    middle is something a reader can see rather than something they have to suspect.

    Deliberately the standard library's `wave` rather than `soundfile`: the frontend
    image does not install the torch or audio stack, and a readout is not worth putting
    libsndfile in it. `st.audio_input` writes WAV, which is the case worth describing; an
    uploaded mp3 falls through to its size, which is still more than nothing.
    """
    with contextlib.suppress(Exception), wave.open(io.BytesIO(clip_bytes)) as handle:
        seconds = handle.getnframes() / handle.getframerate()
        return (
            f"Sending {len(clip_bytes) / 1024:.0f} KB — {seconds:.1f}s at "
            f"{handle.getframerate() / 1000:.1f} kHz, {handle.getnchannels()} channel(s)"
        )
    return f"Sending {len(clip_bytes) / 1024:.0f} KB"


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
        clip_bytes = clip.getvalue() if clip is not None else None

        # `st.audio_input` renders its own player as soon as a recording exists, so adding
        # another one here gave the page two players and a mic button for one clip. The
        # uploader has no player of its own, so that is the only case that needs one -- and
        # it is handed bytes rather than the file object, because a widget that has already
        # read the handle leaves its position at EOF and the second reader gets a short
        # clip or none at all.
        if clip is emotion_upload and clip_bytes is not None:
            st.audio(clip_bytes)

        if clip_bytes is not None:
            # What the service will actually receive, decoded the way it decodes it. A clip
            # that arrives truncated is invisible otherwise: the player shows the file the
            # browser wrote, not the audio the model scores.
            st.caption(_describe(clip_bytes))

        if clip_bytes is not None and st.button("Predict emotion", type="primary"):
            # 120s: this one loads Whisper and runs it per request.
            result = show(
                call(
                    SERVICE,
                    "/predict",
                    {"audio_data": bytes_to_b64(clip_bytes)},
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
