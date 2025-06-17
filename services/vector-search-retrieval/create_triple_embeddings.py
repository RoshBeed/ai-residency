import json
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import glob
import os
import redis
import numpy as np

# Redis Cloud connection (replace with your actual credentials or use environment variables)
REDIS_HOST = 'your-redis-host'
REDIS_PORT = 12345  # your-redis-port
REDIS_PASSWORD = 'your-redis-password'
INDEX_NAME = 'doc_index'
VECTOR_DIM = 128  # Change if your embedding size is different

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False  # binary-safe
)

def load_latest_checkpoint():
    """Load the latest CBOW model checkpoint."""
    print("Loading latest CBOW checkpoint...")
    checkpoint_files = glob.glob('cbow/checkpoints/*.pth')
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in cbow/checkpoints/")
    
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"Using checkpoint: {latest_checkpoint}")
    
    # Load the model state
    state_dict = torch.load(latest_checkpoint)
    return state_dict

def load_tokenizer():
    """Load the CBOW tokenizer mappings."""
    print("Loading tokenizer...")
    with open('cbow/tkn_words_to_ids.pkl', 'rb') as f:
        words_to_ids = pickle.load(f)
    with open('cbow/tkn_ids_to_words.pkl', 'rb') as f:
        ids_to_words = pickle.load(f)
    return words_to_ids, ids_to_words

def load_tokenized_triples():
    """Load the tokenized triples."""
    print("Loading tokenized triples...")
    with open('tokenized_triples.json', 'r') as f:
        data = json.load(f)
    return data

def create_embedding_layer(state_dict, vocab_size, embedding_dim=128):
    """Create embedding layer from CBOW weights."""
    embedding = nn.Embedding(vocab_size, embedding_dim)
    # Extract embedding weights from state dict
    embedding.weight.data.copy_(state_dict['emb.weight'])
    # Freeze the embeddings
    embedding.weight.requires_grad = False
    return embedding

def average_pool(tokens, embedding_layer):
    """Create average pooled vector for a list of tokens."""
    # Convert tokens to tensor
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    # Get embeddings
    embeddings = embedding_layer(tokens_tensor)
    # Average the embeddings
    return torch.mean(embeddings, dim=0).detach().numpy()

def save_doc_embedding_to_redis(doc_id, embedding, text):
    # Save as a Redis hash for vector search
    r.hset(doc_id, mapping={
        'embedding': embedding.astype(np.float32).tobytes(),
        'text': text,
        'doc_id': doc_id
    })

    # Optionally, you can print or log
    # print(f"Saved doc {doc_id} to Redis.")

def process_triples(data, embedding_layer):
    """Process triples and create average pooled vectors. Save positive doc embeddings to Redis."""
    processed_data = {
        'train': [],
        'validation': [],
        'test': []
    }
    doc_counter = 0
    for split in ['train', 'validation', 'test']:
        print(f"\nProcessing {split} split...")
        for triple in tqdm(data[split]):
            # Get average pooled vectors
            query_vector = average_pool(triple['query_tokens'], embedding_layer)
            pos_doc_vector = average_pool(triple['positive_document_tokens'], embedding_layer)
            neg_doc_vector = average_pool(triple['negative_document_tokens'], embedding_layer)

            # Save positive doc embedding to Redis
            doc_id = f"doc:{doc_counter}"
            save_doc_embedding_to_redis(doc_id, pos_doc_vector, triple['positive_document'])
            doc_counter += 1

            processed_data[split].append({
                'query_vector': query_vector.tolist(),
                'positive_document_vector': pos_doc_vector.tolist(),
                'negative_document_vector': neg_doc_vector.tolist(),
                'query': triple['query'],  # Keep original text for reference
                'positive_document': triple['positive_document'],
                'negative_document': triple['negative_document']
            })
    return processed_data

def main():
    # Load data and model
    state_dict = load_latest_checkpoint()
    words_to_ids, ids_to_words = load_tokenizer()
    data = load_tokenized_triples()
    
    # Create embedding layer from CBOW weights
    vocab_size = len(words_to_ids)
    embedding_layer = create_embedding_layer(state_dict, vocab_size)
    
    # Process triples
    processed_data = process_triples(data, embedding_layer)
    
    # Save processed data
    print("\nSaving processed data...")
    with open('triple_embeddings_cbow.json', 'w') as f:
        json.dump(processed_data, f)
    
    # Print statistics
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()} split:")
        print(f"Number of processed triples: {len(processed_data[split])}")
        if processed_data[split]:
            sample = processed_data[split][0]
            print("\nSample vector shapes:")
            print("Query vector shape:", len(sample['query_vector']))
            print("Positive doc vector shape:", len(sample['positive_document_vector']))
            print("Negative doc vector shape:", len(sample['negative_document_vector']))

if __name__ == "__main__":
    main() 