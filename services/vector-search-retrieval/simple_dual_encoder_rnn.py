import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import glob
import os
import json
import wandb

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Load the tokenizer (vocab)
        with open('cbow/tkn_words_to_ids.pkl', 'rb') as f:
            words_to_ids = pickle.load(f)
        vocab_size = len(words_to_ids)
        embedding_dim = 128  # Use the dimension you trained with

        # Find the latest CBOW checkpoint
        checkpoint_files = glob.glob('cbow/checkpoints/*.pth')
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        state_dict = torch.load(latest_checkpoint)

        # Create the embedding layer and load weights
        embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        embedding_layer.weight.data.copy_(state_dict['emb.weight'])
        embedding_layer.weight.requires_grad = False  # freeze weights

        class QryTower(nn.Module):
            def __init__(self, embedding_layer, hidden_size):
                super().__init__()
                self.embedding = embedding_layer
                self.embedding.weight.requires_grad = False
                self.rnn = nn.GRU(input_size=self.embedding.embedding_dim, hidden_size=hidden_size, batch_first=True)

            def forward(self, x):
                if not x:
                    return None
                x = torch.tensor(x, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
                embeds = self.embedding(x)  # (1, seq_len, emb_dim)
                _, h_n = self.rnn(embeds)  # h_n: (1, batch, hidden_size)
                return h_n.squeeze(0).squeeze(0)  # (hidden_size,)

        class DocTower(nn.Module):
            def __init__(self, embedding_layer, hidden_size):
                super().__init__()
                self.embedding = embedding_layer
                self.embedding.weight.requires_grad = False
                self.rnn = nn.GRU(input_size=self.embedding.embedding_dim, hidden_size=hidden_size, batch_first=True)

            def forward(self, x):
                if not x:
                    return None
                x = torch.tensor(x, dtype=torch.long).unsqueeze(0)
                embeds = self.embedding(x)
                _, h_n = self.rnn(embeds)
                return h_n.squeeze(0).squeeze(0)

        qryTower = QryTower(embedding_layer, config.hidden_size)
        docTower = DocTower(embedding_layer, config.hidden_size)

        # Load real tokenized triples (small subset for speed)
        with open('tokenized_triples.json', 'r') as f:
            triples_data = json.load(f)

        # TRAINING LOOP
        params = list(qryTower.rnn.parameters()) + list(docTower.rnn.parameters())
        optimizer = torch.optim.Adam(params, lr=config.learning_rate)
        num_epochs = config.num_epochs
        num_triples = config.num_triples
        margin = config.margin
        print(f"\nTraining on first {num_triples} real triples from the train split with RNN towers for {num_epochs} epochs:\n")
        for epoch in range(num_epochs):
            total_loss = 0
            count = 0
            for triple in triples_data['train'][:num_triples]:
                qry_tokens = triple['query_tokens']
                pos_tokens = triple['positive_document_tokens']
                neg_tokens = triple['negative_document_tokens']

                qry = qryTower(qry_tokens)
                pos = docTower(pos_tokens)
                neg = docTower(neg_tokens)

                if qry is not None and pos is not None and neg is not None:
                    dst_pos = F.cosine_similarity(qry.unsqueeze(0), pos.unsqueeze(0))
                    dst_neg = F.cosine_similarity(qry.unsqueeze(0), neg.unsqueeze(0))
                    dst_mrg = torch.tensor(margin)
                    loss = torch.max(torch.tensor(0.0), dst_mrg - (dst_pos - dst_neg))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    count += 1
            avg_loss = total_loss / max(count,1)
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
            wandb.log({'epoch': epoch+1, 'avg_loss': avg_loss})

        # EVALUATE ON 5 EXAMPLES
        print("\nEvaluating on first 5 real triples after training:\n")
        for i, triple in enumerate(triples_data['train'][:5]):
            qry_tokens = triple['query_tokens']
            pos_tokens = triple['positive_document_tokens']
            neg_tokens = triple['negative_document_tokens']
            qry_text = triple['query']
            pos_text = triple['positive_document']
            neg_text = triple['negative_document']

            qry = qryTower(qry_tokens)
            pos = docTower(pos_tokens)
            neg = docTower(neg_tokens)

            if qry is not None and pos is not None and neg is not None:
                dst_pos = F.cosine_similarity(qry.unsqueeze(0), pos.unsqueeze(0))
                dst_neg = F.cosine_similarity(qry.unsqueeze(0), neg.unsqueeze(0))
                dst_mrg = torch.tensor(margin)
                loss = torch.max(torch.tensor(0.0), dst_mrg - (dst_pos - dst_neg))
                print(f"Example {i+1}:")
                print(f"Query: {qry_text}")
                print(f"Positive doc: {pos_text[:100]}...")
                print(f"Negative doc: {neg_text[:100]}...")
                print(f"Cosine similarity (pos): {dst_pos.item():.4f}")
                print(f"Cosine similarity (neg): {dst_neg.item():.4f}")
                print(f"Triplet loss: {loss.item():.4f}\n")
            else:
                print(f"Example {i+1}: One of the inputs was empty, skipping this triple.\n")

if __name__ == "__main__":
    sweep_config = {
        'method': 'random',
        'parameters': {
            'learning_rate': {'values': [0.001, 0.0005, 0.0001]},
            'hidden_size': {'values': [64, 128, 256]},
            'margin': {'values': [0.1, 0.2, 0.3]},
            'num_epochs': {'value': 10},
            'num_triples': {'value': 500}
        }
    }
    # To run a sweep, comment out the next line and use wandb agent
    train() 