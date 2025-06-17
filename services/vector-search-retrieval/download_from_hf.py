import os
from huggingface_hub import hf_hub_download, snapshot_download

def download_from_huggingface(repo_name, token):
    """
    Download model checkpoints, embeddings, and all intermediary files from Hugging Face Hub.
    
    Args:
        repo_name (str): Name of the repository on Hugging Face
        token (str): Hugging Face API token
    """
    # Create necessary directories
    os.makedirs('cbow/checkpoints', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('vocabulary', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('src', exist_ok=True)

    # Download CBOW checkpoints
    try:
        cbow_files = snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            token=token,
            local_dir="cbow/checkpoints",
            allow_patterns="cbow/checkpoints/*.pth"
        )
        print("Downloaded CBOW checkpoints")
    except Exception as e:
        print(f"Error downloading CBOW checkpoints: {e}")

    # Download main checkpoints
    try:
        main_files = snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            token=token,
            local_dir="checkpoints",
            allow_patterns="checkpoints/*.pth"
        )
        print("Downloaded main checkpoints")
    except Exception as e:
        print(f"Error downloading main checkpoints: {e}")

    # Download raw and intermediary data files
    data_files = [
        'tokenized_triples.json',
        'triples_small.json',
        'extracted_data.json',
        'corpus.pkl',
        'text8'
    ]
    
    for data_file in data_files:
        try:
            hf_hub_download(
                repo_id=repo_name,
                repo_type="model",
                token=token,
                filename=f"data/{data_file}",
                local_dir="."
            )
            print(f"Downloaded {data_file}")
        except Exception as e:
            print(f"Error downloading {data_file}: {e}")

    # Download vocabulary files
    try:
        vocab_files = snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            token=token,
            local_dir="cbow",
            allow_patterns="vocabulary/*.pkl"
        )
        print("Downloaded vocabulary files")
    except Exception as e:
        print(f"Error downloading vocabulary files: {e}")

    # Download configuration files
    config_files = ['sweep.yaml', 'requirements.txt']
    for config_file in config_files:
        try:
            hf_hub_download(
                repo_id=repo_name,
                repo_type="model",
                token=token,
                filename=f"config/{config_file}",
                local_dir="."
            )
            print(f"Downloaded {config_file}")
        except Exception as e:
            print(f"Error downloading {config_file}: {e}")

    # Download source code files
    try:
        code_files = snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            token=token,
            local_dir=".",
            allow_patterns="src/*.py"
        )
        print("Downloaded source code files")
    except Exception as e:
        print(f"Error downloading source code files: {e}")

    print("\nDownload complete! Files are ready for training.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download model files from Hugging Face Hub')
    parser.add_argument('--repo_name', type=str, required=True, help='Name of the repository on Hugging Face')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
    args = parser.parse_args()
    
    download_from_huggingface(args.repo_name, args.token) 