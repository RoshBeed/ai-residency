import torch
import whisper

def load_whisper_model(model_name="tiny"):
    """Load and return the Whisper model and device."""
    model = whisper.load_model(model_name)
    device = model.device
    return model, device

def get_tokenizer():
    """Return the multilingual tokenizer."""
    return whisper.tokenizer.get_tokenizer(multilingual=True)

def preprocess_audio(audio_path):
    """Load and preprocess audio file for Whisper."""
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel

def decode_audio(model, device, audio_path):
    """Transcribe audio using the model and return the decoded text."""
    audio = preprocess_audio(audio_path)
    mel = audio.to(device)
    opt = whisper.DecodingOptions()
    result = whisper.decode(model, mel, opt)
    return result.text

def create_tokens(tokenizer, text):
    """Create token sequence for training."""
    ids = []
    ids += [tokenizer.sot]
    ids += [tokenizer.language_token]
    ids += [tokenizer.transcribe]
    ids += tokenizer.encode(text)
    ids += [tokenizer.eot]
    return ids

def prepare_training_data(tokenizer, device, audio_path, text):
    """Prepare data for training."""
    audio = preprocess_audio(audio_path)
    mel = audio.unsqueeze(0).to(device)
    ids = create_tokens(tokenizer, text)
    tokens = torch.tensor(ids).unsqueeze(0).to(device)
    return tokens, mel

def forward(model, tokens, mel):
    """Forward pass through the model."""
    return model(tokens=tokens, mel=mel)

def decode_tokens(tokenizer, token_ids):
    """Decode token IDs back to text."""
    return tokenizer.decode(token_ids) 