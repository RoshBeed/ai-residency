from datasets import load_dataset
import json

def extract_queries_and_documents():
    # Load the dataset
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")
    
    # Dictionary to store our extracted data
    extracted_data = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    # Extract data from each split
    for split in ['train', 'validation', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Process each example
        for example in dataset[split]:
            # Extract query
            query = example['query']
            
            # Extract passages and their relevance labels
            passages = example['passages']['passage_text']
            relevance_labels = example['passages']['is_selected']  # 1 if relevant, 0 if not
            
            # Create list of (passage, relevance) pairs
            passage_relevance_pairs = list(zip(passages, relevance_labels))
            
            # Store the query and its passages with relevance
            extracted_data[split].append({
                'query': query,
                'passages_with_relevance': [
                    {
                        'passage': passage,
                        'is_relevant': bool(is_relevant)  # Convert to boolean for clarity
                    }
                    for passage, is_relevant in passage_relevance_pairs
                ]
            })
            
            # Print progress every 1000 examples
            if len(extracted_data[split]) % 1000 == 0:
                print(f"Processed {len(extracted_data[split])} examples")
    
    # Save the extracted data
    print("\nSaving extracted data...")
    with open('extracted_data.json', 'w') as f:
        json.dump(extracted_data, f, indent=2)
    
    # Print some statistics
    print("\nExtraction complete!")
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()} split:")
        print(f"Number of queries: {len(extracted_data[split])}")
        
        # Calculate relevance statistics
        total_passages = 0
        relevant_passages = 0
        for item in extracted_data[split]:
            for passage_info in item['passages_with_relevance']:
                total_passages += 1
                if passage_info['is_relevant']:
                    relevant_passages += 1
        
        print(f"Total number of passages: {total_passages}")
        print(f"Number of relevant passages: {relevant_passages}")
        print(f"Percentage of relevant passages: {(relevant_passages/total_passages)*100:.2f}%")
        
        # Show a sample
        if extracted_data[split]:
            sample = extracted_data[split][0]
            print("\nSample query:", sample['query'])
            print("Number of passages:", len(sample['passages_with_relevance']))
            print("\nSample passages with relevance:")
            for i, passage_info in enumerate(sample['passages_with_relevance'][:2]):  # Show first 2 passages
                print(f"\nPassage {i+1}:")
                print(f"Relevance: {'Relevant' if passage_info['is_relevant'] else 'Not Relevant'}")
                print(f"Preview: {passage_info['passage'][:200]}...")

if __name__ == "__main__":
    extract_queries_and_documents() 