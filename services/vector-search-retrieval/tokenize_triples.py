import json
import pickle
from tqdm import tqdm
import numpy as np

def load_tokenizer():
    """Load the CBOW tokenizer mappings."""
    with open('tkn_words_to_ids.pkl', 'rb') as f:
        words_to_ids = pickle.load(f)
    with open('tkn_ids_to_words.pkl', 'rb') as f:
        ids_to_words = pickle.load(f)
    return words_to_ids, ids_to_words

def tokenize_text(text, words_to_ids):
    """Tokenize text using the CBOW tokenizer."""
    # Convert to lowercase and split
    words = text.lower().split()
    # Convert words to IDs, using 0 for unknown words
    token_ids = [words_to_ids.get(word, 0) for word in words]
    return token_ids

def process_triples(input_file, output_file):
    """Process triples and tokenize queries and documents."""
    print("Loading tokenizer...")
    words_to_ids, ids_to_words = load_tokenizer()
    
    print("Loading triples...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    tokenized_data = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    for split in ['train', 'validation', 'test']:
        print(f"\nTokenizing {split} split...")
        for triple in tqdm(data[split]):
            query = triple['query']
            pos_doc = triple['positive_doc']
            neg_doc = triple['negative_doc']
            
            # Tokenize query and documents
            query_tokens = tokenize_text(query, words_to_ids)
            pos_doc_tokens = tokenize_text(pos_doc, words_to_ids)
            neg_doc_tokens = tokenize_text(neg_doc, words_to_ids)
            
            tokenized_data[split].append({
                'query_tokens': query_tokens,
                'positive_document_tokens': pos_doc_tokens,
                'negative_document_tokens': neg_doc_tokens,
                'query': query,  # Keep original text for reference
                'positive_document': pos_doc,
                'negative_document': neg_doc
            })
    
    print("Saving tokenized triples...")
    with open(output_file, 'w') as f:
        json.dump(tokenized_data, f, indent=2)
    
    # Print statistics
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()} split:")
        print(f"Number of tokenized triples: {len(tokenized_data[split])}")
        if tokenized_data[split]:
            sample = tokenized_data[split][0]
            print("\nSample tokenized triple:")
            print("Query tokens length:", len(sample['query_tokens']))
            print("Positive doc tokens length:", len(sample['positive_document_tokens']))
            print("Negative doc tokens length:", len(sample['negative_document_tokens']))

if __name__ == "__main__":
    input_file = "triples_small.json"
    output_file = "tokenized_triples.json"
    process_triples(input_file, output_file) 