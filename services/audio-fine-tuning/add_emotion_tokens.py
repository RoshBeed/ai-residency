#!/usr/bin/env python3
"""
Add special emotion tokens to the Whisper tokenizer and save it locally.
"""
from transformers import WhisperTokenizer

# List of special emotion tokens
emotion_tokens = [
    '<emotion_happy>',
    '<emotion_sad>',
    '<emotion_fearful>',
    '<emotion_neutral>',
    '<emotion_angry>',
    '<emotion_disgust>'
]

# Load the base Whisper tokenizer
model_name = 'openai/whisper-small'
tokenizer = WhisperTokenizer.from_pretrained(model_name)

# Add special tokens
special_tokens_dict = {'additional_special_tokens': emotion_tokens}
num_added = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added} special tokens: {emotion_tokens}")

# Save the updated tokenizer
save_dir = './whisper_tokenizer_emotion'
tokenizer.save_pretrained(save_dir)
print(f"Saved updated tokenizer to {save_dir}") 