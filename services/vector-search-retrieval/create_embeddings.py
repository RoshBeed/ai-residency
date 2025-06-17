import json
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn

def load_tokenizer():
    """Load the CBOW tokenizer mappings."""
    print("Loading tokenizer...")
    with open('tkn_words_to_ids.pkl', 'rb') as f:
        words_to_ids = pickle.load(f)
    with open('tkn_ids_to_words.pkl', 'rb') as f:
        ids_to_words = pickle.load(f)
    return words_to_ids, ids_to_words

def load_tokenized_triples():
    """Load the tokenized triples."""
    print("Loading tokenized triples...")
    with open('tokenized_triples.json', 'r') as f:
        data = json.load(f)
    return data

def create_embedding_layer(vocab_size, embedding_dim=128):
    """Create a simple embedding layer."""
    embedding = nn.Embedding(vocab_size, embedding_dim)
    # Initialize with random weights
    nn.init.xavier_uniform_(embedding.weight)
    return embedding

def average_pool(tokens, embedding_layer):
    """Create average pooled vector for a list of tokens."""
    # Convert tokens to tensor
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    # Get embeddings
    embeddings = embedding_layer(tokens_tensor)
    # Average the embeddings and detach before converting to numpy
    return torch.mean(embeddings, dim=0).detach().numpy()

def process_triples(data, embedding_layer):
    """Process triples and create average pooled vectors."""
    processed_data = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    for split in ['train', 'validation', 'test']:
        print(f"\nProcessing {split} split...")
        for triple in tqdm(data[split]):
            # Get average pooled vectors
            query_vector = average_pool(triple['query_tokens'], embedding_layer)
            pos_doc_vector = average_pool(triple['positive_document_tokens'], embedding_layer)
            neg_doc_vector = average_pool(triple['negative_document_tokens'], embedding_layer)
            
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
    # Load data
    words_to_ids, ids_to_words = load_tokenizer()
    data = load_tokenized_triples()
    
    # Create embedding layer
    vocab_size = len(words_to_ids)
    embedding_layer = create_embedding_layer(vocab_size)
    
    # Process triples
    processed_data = process_triples(data, embedding_layer)
    
    # Save processed data
    print("\nSaving processed data...")
    with open('triple_embeddings.json', 'w') as f:
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