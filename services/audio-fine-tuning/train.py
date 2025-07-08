import torch
from model import prepare_training_data, forward

def train(model, tokenizer, device, audio_path, target_text, lr=1e-4):
    """Fine-tune the model on a single (audio, text) pair."""
    tokens, mel = prepare_training_data(tokenizer, device, audio_path, target_text)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    pred = forward(model, tokens, mel)
    trgt = tokens[:, 1:].contiguous()
    pred = pred[:, :-1, :].contiguous()
    loss = criterion(pred.transpose(1, 2), trgt)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model

if __name__ == "__main__":
    model, device = load_whisper_model("tiny")
    tokenizer = get_tokenizer()
    train(model, tokenizer, device) 