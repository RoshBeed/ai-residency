import os
from huggingface_hub import hf_hub_download, list_repo_files
import logging
import json
import pickle
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_file_integrity(file_path):
    """Verify that a file exists and can be read."""
    if not os.path.exists(file_path):
        return False, f"File {file_path} does not exist"
    
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                json.load(f)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                pickle.load(f)
        elif file_path.endswith('.pth'):
            torch.load(file_path, map_location=torch.device('cpu'))
        return True, "File is valid"
    except Exception as e:
        return False, f"Error reading file {file_path}: {str(e)}"

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

    # Required files for training
    required_files = {
        'vocabulary': 'cbow/tkn_words_to_ids.pkl',
        'checkpoint': 'cbow/checkpoints/*.pth',
        'data': 'tokenized_triples.json',
        'config': 'sweep.yaml'
    }

    # List all files in the repository
    try:
        all_files = list_repo_files(repo_id=repo_name, repo_type="model", token=token)
        logger.info("Found files in repository:")
        for file in all_files:
            logger.info(f"- {file}")
    except Exception as e:
        logger.error(f"Error listing repository files: {e}")
        return

    # Download vocabulary file
    vocab_file = required_files['vocabulary']
    try:
        logger.info(f"Downloading {vocab_file}...")
        hf_hub_download(
            repo_id=repo_name,
            repo_type="model",
            token=token,
            filename=vocab_file,
            local_dir="."
        )
        is_valid, message = verify_file_integrity(vocab_file)
        if is_valid:
            logger.info(f"Successfully downloaded and verified {vocab_file}")
        else:
            logger.error(message)
    except Exception as e:
        logger.error(f"Error downloading {vocab_file}: {e}")

    # Download CBOW checkpoints
    cbow_checkpoint_files = [f for f in all_files if f.startswith("cbow/checkpoints/") and f.endswith(".pth")]
    for checkpoint_file in cbow_checkpoint_files:
        try:
            logger.info(f"Downloading {checkpoint_file}...")
            hf_hub_download(
                repo_id=repo_name,
                repo_type="model",
                token=token,
                filename=checkpoint_file,
                local_dir="."
            )
            is_valid, message = verify_file_integrity(checkpoint_file)
            if is_valid:
                logger.info(f"Successfully downloaded and verified {checkpoint_file}")
            else:
                logger.error(message)
        except Exception as e:
            logger.error(f"Error downloading {checkpoint_file}: {e}")

    # Download tokenized triples
    data_file = required_files['data']
    try:
        logger.info(f"Downloading {data_file}...")
        hf_hub_download(
            repo_id=repo_name,
            repo_type="model",
            token=token,
            filename=data_file,
            local_dir="."
        )
        is_valid, message = verify_file_integrity(data_file)
        if is_valid:
            logger.info(f"Successfully downloaded and verified {data_file}")
        else:
            logger.error(message)
    except Exception as e:
        logger.error(f"Error downloading {data_file}: {e}")

    # Download configuration
    config_file = required_files['config']
    try:
        logger.info(f"Downloading {config_file}...")
        hf_hub_download(
            repo_id=repo_name,
            repo_type="model",
            token=token,
            filename=config_file,
            local_dir="."
        )
        logger.info(f"Successfully downloaded {config_file}")
    except Exception as e:
        logger.error(f"Error downloading {config_file}: {e}")

    # Verify all required files are present and valid
    missing_files = []
    invalid_files = []
    for file in required_files.values():
        if '*' in file:  # Handle glob patterns
            pattern = file.replace('*', '')
            matching_files = [f for f in os.listdir(os.path.dirname(file)) if pattern in f]
            if not matching_files:
                missing_files.append(file)
            else:
                for matching_file in matching_files:
                    full_path = os.path.join(os.path.dirname(file), matching_file)
                    is_valid, message = verify_file_integrity(full_path)
                    if not is_valid:
                        invalid_files.append(f"{matching_file}: {message}")
        else:
            if not os.path.exists(file):
                missing_files.append(file)
            else:
                is_valid, message = verify_file_integrity(file)
                if not is_valid:
                    invalid_files.append(f"{file}: {message}")

    if missing_files:
        logger.error("\nMissing files:")
        for file in missing_files:
            logger.error(f"- {file}")
    
    if invalid_files:
        logger.error("\nInvalid files:")
        for file in invalid_files:
            logger.error(f"- {file}")
    
    if not missing_files and not invalid_files:
        logger.info("\nAll required files were successfully downloaded and verified!")

    logger.info("\nDownload complete! Files are ready for training.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download model files from Hugging Face Hub')
    parser.add_argument('--repo_name', type=str, required=True, help='Name of the repository on Hugging Face')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
    args = parser.parse_args()
    
    download_from_huggingface(args.repo_name, args.token)