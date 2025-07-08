from model import load_whisper_model, get_tokenizer, decode_audio
from train import train

def main():
    audio_path = "audio.mp3"
    target_text = "My name is Rosh"

    model, device = load_whisper_model("tiny")
    tokenizer = get_tokenizer()

    base_output = decode_audio(model, device, audio_path)
    model = train(model, tokenizer, device, audio_path, target_text)
    fine_tuned_output = decode_audio(model, device, audio_path)

    print("Base model output:", base_output)
    print("Fine-tuned model output:", fine_tuned_output)

if __name__ == "__main__":
    main()

