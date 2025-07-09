#!/usr/bin/env python3
"""
Download and prepare speech emotion datasets for Whisper fine-tuning.
"""

import os
import requests
import zipfile
import tarfile
import subprocess
import shutil
from pathlib import Path

def download_file(url, filename):
    """Download a file with progress indicator."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end='', flush=True)
    print(f"\nDownloaded {filename}")

def setup_ravdess():
    """Setup RAVDESS dataset structure."""
    dataset_dir = Path("datasets/ravdess")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("Setting up RAVDESS dataset structure...")
    print("Note: Full RAVDESS dataset requires registration.")
    
    # Create emotion directories
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    for emotion in emotions:
        emotion_dir = dataset_dir / emotion
        emotion_dir.mkdir(exist_ok=True)
        print(f"Created directory: {emotion_dir}")
    
    # Create a metadata file
    metadata = {
        "dataset": "RAVDESS",
        "description": "Ryerson Audio-Visual Database of Emotional Speech and Song",
        "emotions": emotions,
        "total_files": 0,
        "format": "wav",
        "sample_rate": 16000,
        "download_url": "https://zenodo.org/record/1188976"
    }
    
    import json
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset structure created at: {dataset_dir}")
    print("\nTo get the full RAVDESS dataset:")
    print("1. Visit: https://zenodo.org/record/1188976")
    print("2. Register and download the dataset")
    print("3. Extract and organize files by emotion")
    
    return dataset_dir

def setup_crema_d():
    """Setup CREMA-D dataset with Git LFS."""
    print("\nSetting up CREMA-D dataset...")
    
    dataset_dir = Path("datasets/crema_d")
    
    # Check if git-lfs is installed
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git LFS is not installed. Installing...")
        try:
            subprocess.run(["brew", "install", "git-lfs"], check=True)
            subprocess.run(["git", "lfs", "install"], check=True)
        except subprocess.CalledProcessError:
            print("Failed to install Git LFS. Please install it manually:")
            print("brew install git-lfs")
            print("git lfs install")
            return None
    
    # Remove existing directory if it exists
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    
    # Clone CREMA-D with Git LFS
    print("Cloning CREMA-D repository with Git LFS...")
    try:
        subprocess.run([
            "git", "lfs", "clone", 
            "https://github.com/CheyneyComputerScience/CREMA-D.git",
            str(dataset_dir)
        ], check=True)
        
        # Move files from CREMA-D subdirectory
        crema_subdir = dataset_dir / "CREMA-D"
        if crema_subdir.exists():
            for item in crema_subdir.iterdir():
                shutil.move(str(item), str(dataset_dir))
            crema_subdir.rmdir()
        
        print(f"CREMA-D dataset ready at: {dataset_dir}")
        return dataset_dir
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone CREMA-D: {e}")
        print("Trying alternative GitLab mirror...")
        
        try:
            subprocess.run([
                "git", "lfs", "clone",
                "https://gitlab.com/cs-cooper-lab/crema-d-mirror.git",
                str(dataset_dir)
            ], check=True)
            
            print(f"CREMA-D dataset ready at: {dataset_dir}")
            return dataset_dir
            
        except subprocess.CalledProcessError as e2:
            print(f"Failed to clone from GitLab mirror: {e2}")
            return None

def setup_emodb():
    """Setup EmoDB dataset (alternative)."""
    print("\nSetting up EmoDB dataset...")
    
    dataset_dir = Path("datasets/emodb")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # EmoDB is available via direct download
    emodb_url = "http://emodb.bilderbar.info/download/download.zip"
    
    try:
        zip_file = dataset_dir / "emodb.zip"
        download_file(emodb_url, zip_file)
        
        # Extract
        print("Extracting EmoDB...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # Clean up zip file
        zip_file.unlink()
        
        print(f"EmoDB dataset ready at: {dataset_dir}")
        return dataset_dir
        
    except Exception as e:
        print(f"Failed to download EmoDB: {e}")
        return None

def create_sample_dataset():
    """Create a sample dataset structure for testing."""
    print("\nCreating sample dataset structure...")
    
    sample_dir = Path("datasets/sample_emotion")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    for emotion in emotions:
        emotion_dir = sample_dir / emotion
        emotion_dir.mkdir(exist_ok=True)
        print(f"Created directory: {emotion_dir}")
    
    # Create metadata
    metadata = {
        "dataset": "Sample Emotion Dataset",
        "description": "Sample structure for emotion detection fine-tuning",
        "emotions": emotions,
        "total_files": 0,
        "format": "wav",
        "sample_rate": 16000,
        "note": "This is a sample structure. Add your audio files to each emotion directory."
    }
    
    import json
    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Sample dataset structure created at: {sample_dir}")
    return sample_dir

def main():
    """Main function to setup datasets."""
    print("Speech Emotion Dataset Setup")
    print("=" * 40)
    
    # Setup RAVDESS structure
    ravdess_dir = setup_ravdess()
    
    # Setup CREMA-D
    crema_dir = setup_crema_d()
    
    # Setup EmoDB as another alternative
    emodb_dir = setup_emodb()
    
    # Create sample structure
    sample_dir = create_sample_dataset()
    
    print("\n" + "=" * 40)
    print("Dataset setup complete!")
    print(f"RAVDESS structure: {ravdess_dir}")
    print(f"CREMA-D dataset: {crema_dir}")
    print(f"EmoDB dataset: {emodb_dir}")
    print(f"Sample structure: {sample_dir}")
    
    print("\nRecommended next steps:")
    print("1. For immediate testing: Use the sample structure and add your own audio files")
    print("2. For full dataset: Download RAVDESS from https://zenodo.org/record/1188976")
    print("3. For public dataset: Use CREMA-D (if successfully downloaded)")
    print("4. Organize audio files by emotion in the created directories")
    
    if crema_dir and crema_dir.exists():
        print(f"\nCREMA-D successfully downloaded! Check: {crema_dir}")
        print("Audio files are in AudioWAV/ directory")
    else:
        print("\nCREMA-D download failed. You can:")
        print("- Install Git LFS manually and retry")
        print("- Use the sample structure for testing")
        print("- Download RAVDESS manually")

if __name__ == "__main__":
    main() 