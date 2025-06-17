from datasets import load_dataset
import random
import json
from tqdm import tqdm

def generate_triples(max_examples_per_split=100):  # Added parameter to limit examples
    # Load the dataset
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")
    
    # Dictionary to store our triples
    triples = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    # Process each split
    for split in ['train', 'validation', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Take only a subset of examples for testing
        split_data = dataset[split].select(range(min(max_examples_per_split, len(dataset[split]))))
        
        # First, collect all passages for negative sampling
        all_passages = []
        for example in split_data:
            passages = example['passages']['passage_text']
            all_passages.extend(passages)
        all_passages = list(set(all_passages))  # Remove duplicates
        print(f"Total unique passages for negative sampling: {len(all_passages)}")
        
        # Generate triples
        for example in tqdm(split_data, desc=f"Generating triples for {split}"):
            query = example['query']
            
            # Get relevant passages
            passages = example['passages']['passage_text']
            relevance = example['passages']['is_selected']
            
            # For each relevant passage, create a triple
            for i, (passage, is_relevant) in enumerate(zip(passages, relevance)):
                if is_relevant:  # This is a positive document
                    # Sample a negative document
                    negative_passages = [p for p in all_passages if p != passage]
                    if negative_passages:  # Make sure we have negative samples
                        negative_doc = random.choice(negative_passages)
                        
                        # Create the triple
                        triple = {
                            'query': query,
                            'positive_doc': passage,
                            'negative_doc': negative_doc
                        }
                        triples[split].append(triple)
        
        print(f"Generated {len(triples[split])} triples for {split} split")
    
    # Save the triples
    print("\nSaving triples...")
    with open('triples_small.json', 'w') as f:  # Changed filename to indicate it's a small dataset
        json.dump(triples, f, indent=2)
    
    # Print some statistics and examples
    print("\nTriple generation complete!")
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()} split:")
        print(f"Number of triples: {len(triples[split])}")
        
        # Show a sample triple
        if triples[split]:
            sample = triples[split][0]
            print("\nSample triple:")
            print(f"Query: {sample['query']}")
            print(f"\nPositive document: {sample['positive_doc'][:200]}...")
            print(f"\nNegative document: {sample['negative_doc'][:200]}...")

if __name__ == "__main__":
    # Process only 100 examples per split for testing
    generate_triples(max_examples_per_split=100) 