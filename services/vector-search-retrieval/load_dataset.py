from datasets import load_dataset
from collections import Counter

def main():
    # Load the MS MARCO dataset
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")
    
    # Print information about each split
    print("\nDataset splits:")
    print("-" * 50)
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()} split:")
        print(f"Number of examples: {len(dataset[split])}")
        
        # Show multiple examples from each split
        print("\nExamples:")
        for i in range(3):  # Show 3 examples
            example = dataset[split][i]
            print(f"\nExample {i+1}:")
            print(f"Query: {example['query']}")
            print(f"Number of passages: {len(example['passages']['passage_text'])}")
            print(f"First passage preview: {example['passages']['passage_text'][0][:200]}...")
        
        # Calculate some statistics
        query_lengths = [len(ex['query'].split()) for ex in dataset[split]]
        passage_lengths = [len(p.split()) for ex in dataset[split] for p in ex['passages']['passage_text']]
        
        print(f"\nStatistics for {split} split:")
        print(f"Average query length: {sum(query_lengths)/len(query_lengths):.2f} words")
        print(f"Average passage length: {sum(passage_lengths)/len(passage_lengths):.2f} words")
        print(f"Total number of passages: {len(passage_lengths)}")

if __name__ == "__main__":
    main() 