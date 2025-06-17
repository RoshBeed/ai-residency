import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.environ.get('REDIS_HOST', 'your-redis-host')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 12345))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', 'your-redis-password')
INDEX_NAME = 'doc_index'
EMBEDDING_DIM = 128
TOP_K = 5

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False
)

# Ensure RediSearch index exists
try:
    r.execute_command(
        f"FT.INFO {INDEX_NAME}"
    )
    print(f"Index '{INDEX_NAME}' already exists.")
except redis.ResponseError:
    print(f"Creating index '{INDEX_NAME}'...")
    r.execute_command(
        f"FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 doc: SCHEMA embedding VECTOR HNSW 6 TYPE FLOAT32 DIM {EMBEDDING_DIM} DISTANCE_METRIC COSINE text TEXT doc_id TAG"
    )
    print(f"Index '{INDEX_NAME}' created.")

# Load tokenizer
with open('cbow/tkn_words_to_ids.pkl', 'rb') as f:
    words_to_ids = pickle.load(f)
vocab_size = len(words_to_ids)

# Load latest CBOW checkpoint for embedding layer
import glob
checkpoint_files = glob.glob('cbow/checkpoints/*.pth')
latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
state_dict = torch.load(latest_checkpoint, map_location='cpu')

embedding_layer = nn.Embedding(vocab_size, EMBEDDING_DIM)
embedding_layer.weight.data.copy_(state_dict['emb.weight'])
embedding_layer.weight.requires_grad = False

# Define DocTower (same as in save_doc_embeddings_to_redis.py)
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

hidden_size = EMBEDDING_DIM

docTower = DocTower(embedding_layer, hidden_size)
docTower.eval()

# Tokenize query
def tokenize(text):
    return [words_to_ids.get(w, 0) for w in text.strip().split()]

# Interactive query loop
while True:
    query = input("Enter your query (or 'exit' to quit): ").strip()
    if query.lower() == 'exit':
        break
    tokens = tokenize(query)
    with torch.no_grad():
        query_emb = docTower(tokens).detach().cpu().numpy().astype(np.float32)
    # Redis expects bytes for VECTOR field
    query_emb_bytes = query_emb.tobytes()
    # Perform ANN search
    res = r.execute_command(
        "FT.SEARCH",
        INDEX_NAME,
        f"*=>[KNN {TOP_K} @embedding $vec as score]",
        "RETURN", 2, "text", "score",
        "PARAMS", 2, "vec", query_emb_bytes,
        "DIALECT", 2
    )
    if len(res) <= 1:
        print("No results found.")
        continue
    print(f"Top {TOP_K} results:")
    results = []
    # RediSearch result: [count, doc_id1, [fields...], doc_id2, [fields...], ...]
    for rank, i in enumerate(range(1, len(res)-1, 2), 1):
        doc_id = res[i]
        doc_fields = res[i+1]
        if not isinstance(doc_fields, list) or len(doc_fields) < 2:
            continue
        text = None
        score = None
        for j in range(0, len(doc_fields), 2):
            key = doc_fields[j]
            value = doc_fields[j+1]
            if key == b'text':
                text = value.decode('utf-8', errors='ignore')
            elif key == b'score':
                try:
                    score = float(value)
                except Exception:
                    score = None
        if score is not None:
            results.append((score, text if text is not None else '[No text found]'))
    if not results:
        print("[Debug] Raw RediSearch result:", res)
    else:
        # Sort by score (ascending: lower cosine distance = more similar)
        results.sort(key=lambda x: x[0])
        for idx, (score, text) in enumerate(results, 1):
            print(f"Rank {idx}: Score={score:.4f}\n{text}\n---")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device) 