import os
import torch
import glob
from huggingface_hub import HfApi, create_repo
from datetime import datetime

def upload_to_huggingface(repo_name, token):
    """
    Upload model checkpoints, embeddings, and all intermediary files to Hugging Face Hub.
    
    Args:
        repo_name (str): Name of the repository to create/use on Hugging Face
        token (str): Hugging Face API token
    """
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_name, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload CBOW checkpoints
    cbow_checkpoints = glob.glob('cbow/checkpoints/*.pth')
    for checkpoint in cbow_checkpoints:
        print(f"Uploading {checkpoint}...")
        api.upload_file(
            path_or_fileobj=checkpoint,
            path_in_repo=f"cbow/checkpoints/{os.path.basename(checkpoint)}",
            repo_id=repo_name,
            repo_type="model"
        )

    # Upload any model checkpoints from the main checkpoints directory
    main_checkpoints = glob.glob('checkpoints/*.pth')
    for checkpoint in main_checkpoints:
        print(f"Uploading {checkpoint}...")
        api.upload_file(
            path_or_fileobj=checkpoint,
            path_in_repo=f"checkpoints/{os.path.basename(checkpoint)}",
            repo_id=repo_name,
            repo_type="model"
        )

    # Upload raw and intermediary data files
    data_files = [
        'tokenized_triples.json',
        'triples_small.json',
        'extracted_data.json',
        'corpus.pkl',
        'text8'
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"Uploading {data_file}...")
            api.upload_file(
                path_or_fileobj=data_file,
                path_in_repo=f"data/{data_file}",
                repo_id=repo_name,
                repo_type="model"
            )

    # Upload vocabulary and tokenizer files
    vocab_files = glob.glob('cbow/*.pkl')
    for vocab_file in vocab_files:
        print(f"Uploading {vocab_file}...")
        api.upload_file(
            path_or_fileobj=vocab_file,
            path_in_repo=f"vocabulary/{os.path.basename(vocab_file)}",
            repo_id=repo_name,
            repo_type="model"
        )

    # Upload configuration files
    config_files = ['sweep.yaml', 'requirements.txt']
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"Uploading {config_file}...")
            api.upload_file(
                path_or_fileobj=config_file,
                path_in_repo=f"config/{config_file}",
                repo_id=repo_name,
                repo_type="model"
            )

    # Upload source code files
    code_files = glob.glob('*.py')
    for code_file in code_files:
        print(f"Uploading {code_file}...")
        api.upload_file(
            path_or_fileobj=code_file,
            path_in_repo=f"src/{code_file}",
            repo_id=repo_name,
            repo_type="model"
        )

    print(f"\nUpload complete! Files are available at: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Upload model files to Hugging Face Hub')
    parser.add_argument('--repo_name', type=str, required=True, help='Name of the repository on Hugging Face')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
    args = parser.parse_args()
    
    upload_to_huggingface(args.repo_name, args.token) 