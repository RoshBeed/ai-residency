import torch
import torch.nn as nn
import pickle
import glob
import json
import redis
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

# Redis Cloud connection (replace with your actual credentials or use environment variables)
REDIS_HOST = os.environ.get('REDIS_HOST', 'your-redis-host')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 12345))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', 'your-redis-password')
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False  # binary-safe
)

# Load tokenizer
with open('cbow/tkn_words_to_ids.pkl', 'rb') as f:
    words_to_ids = pickle.load(f)
vocab_size = len(words_to_ids)
embedding_dim = 128  # Change if needed

# Load latest CBOW checkpoint for embedding layer
checkpoint_files = glob.glob('cbow/checkpoints/*.pth')
latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
state_dict = torch.load(latest_checkpoint, map_location='cpu')

embedding_layer = nn.Embedding(vocab_size, embedding_dim)
embedding_layer.weight.data.copy_(state_dict['emb.weight'])
embedding_layer.weight.requires_grad = False

# Define DocTower (copy from simple_dual_encoder_rnn.py)
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

# Load doc tower weights (if you saved them separately, load here)
# Otherwise, use the same initialization as in training
hidden_size = 128  # Set to your trained hidden size

docTower = DocTower(embedding_layer, hidden_size)
# Optionally: docTower.load_state_dict(torch.load('doc_tower.pth'))
docTower.eval()

# Load tokenized triples
with open('tokenized_triples.json', 'r') as f:
    triples_data = json.load(f)

# Collect all unique positive documents
seen = set()
documents = []
for split in ['train', 'validation', 'test']:
    for triple in triples_data[split]:
        doc_text = triple['positive_document']
        doc_tokens = tuple(triple['positive_document_tokens'])
        if doc_tokens not in seen:
            seen.add(doc_tokens)
            documents.append((doc_tokens, doc_text))

print(f"Found {len(documents)} unique positive documents.")

def save_doc_embedding_to_redis(doc_id, embedding, text):
    r.hset(doc_id, mapping={
        'embedding': embedding.astype(np.float32).tobytes(),
        'text': text,
        'doc_id': doc_id
    })

# Compute and save embeddings
docTower.eval()
with torch.no_grad():
    for idx, (doc_tokens, doc_text) in enumerate(tqdm(documents, desc='Saving doc embeddings to Redis')):
        embedding = docTower(list(doc_tokens)).detach().cpu().numpy()
        doc_id = f"doc:{idx}"
        save_doc_embedding_to_redis(doc_id, embedding, doc_text)

print(f"Saved {len(documents)} doc embeddings to Redis Cloud.") 