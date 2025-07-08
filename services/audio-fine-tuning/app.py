import streamlit as st
from st_audiorec import st_audiorec
import tempfile
from model import load_whisper_model, decode_audio, get_tokenizer, preprocess_audio
from train import train
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Whisper Fine-tuning Demo", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .stButton>button {font-size: 1.1rem; padding: 0.5em 2em; border-radius: 8px;}
    .stTextInput>div>input {font-size: 1.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0.5em;'>🎤 Whisper Audio Fine-tuning Demo</h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
        1. **Record** your audio.
        2. **Compare** the base and fine-tuned model outputs.
        3. **Correct** the text and fine-tune.
        4. **Repeat** with new audio to see improvement!
        """
    )

if 'finetuned_model' not in st.session_state:
    st.session_state['finetuned_model'] = None
    st.session_state['finetuned_tokenizer'] = None
    st.session_state['finetuned_device'] = None

if 'finetune_history' not in st.session_state:
    st.session_state['finetune_history'] = []

st.markdown("### 1. Record your audio")
with st.container():
    audio_bytes = st_audiorec()
    audio_path = None

    base_output = ""
    ft_output = ""
    mel = None

    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name

        base_model, base_device = load_whisper_model("tiny")
        base_tokenizer = get_tokenizer()
        base_output = decode_audio(base_model, base_device, audio_path)

        if st.session_state['finetuned_model'] is not None:
            ft_model = st.session_state['finetuned_model']
            ft_tokenizer = st.session_state['finetuned_tokenizer']
            ft_device = st.session_state['finetuned_device']
            ft_output = decode_audio(ft_model, ft_device, audio_path)
        else:
            ft_output = "(Fine-tuned model not available yet)"

        mel = preprocess_audio(audio_path)
        st.audio(audio_bytes, format="audio/wav")
    else:
        st.info("Click 'Start Recording', say your phrase, then click 'Stop'.")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🤖 Base Model Output")
    st.info(base_output if audio_bytes else "Record audio to see output.")
with col2:
    st.markdown("#### 🛠️ Fine-tuned Model Output")
    st.success(ft_output if audio_bytes else "Record audio to see output.")
    st.markdown("##### Fine-tune the Model")
    with st.form("finetune_form", clear_on_submit=False):
        correct_text = st.text_input(
            "Enter the correct text for the selected audio:",
            key="correct_text_input"
        )
        fine_tune_btn = st.form_submit_button("🚀 Fine-tune")
        if fine_tune_btn and audio_bytes and mel is not None and correct_text.strip() != "":
            model_to_finetune = st.session_state['finetuned_model'] or base_model
            tokenizer_to_finetune = st.session_state['finetuned_tokenizer'] or base_tokenizer
            device_to_finetune = st.session_state['finetuned_device'] or base_device

            st.session_state['finetune_history'].append({
                "mel": mel.cpu().numpy() if hasattr(mel, 'cpu') else np.array(mel),
                "label": correct_text
            })

            model = train(model_to_finetune, tokenizer_to_finetune, device_to_finetune, audio_path, correct_text)
            st.session_state['finetuned_model'] = model
            st.session_state['finetuned_tokenizer'] = tokenizer_to_finetune
            st.session_state['finetuned_device'] = device_to_finetune
            st.success("Model fine-tuned! Record a new audio clip to compare again.")

# Display fine-tuning history as a compact table
st.markdown("---")
if st.session_state['finetune_history']:
    st.markdown("### 📜 Fine-tuning History")
    for i, entry in enumerate(st.session_state['finetune_history']):
        col1, col2 = st.columns([1, 5])
        with col1:
            fig, ax = plt.subplots(figsize=(2, 0.5))
            ax.imshow(entry['mel'], aspect='auto', origin='lower')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            st.image(buf.getvalue(), use_container_width=True)
        with col2:
            st.markdown(
                f"""
                <div style='
                    display: flex;
                    align-items: center;
                    height: 100%;
                    background: #181818;
                    border-radius: 8px;
                    padding: 0.5em 1em;
                    margin-bottom: 0.5em;
                    font-size: 1.1em;
                '>
                    {entry['label']}
                </div>
                """,
                unsafe_allow_html=True
            ) 