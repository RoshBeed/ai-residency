#!/usr/bin/env python3
"""
Prepare speech emotion datasets for Whisper fine-tuning.
Organizes audio files by emotion and creates training/validation splits.
"""

import os
import shutil
import json
import random
from pathlib import Path
from collections import defaultdict

def organize_crema_d():
    """Organize CREMA-D dataset by emotion."""
    print("Organizing CREMA-D dataset...")
    
    source_dir = Path("datasets/crema_d/AudioWAV")
    target_dir = Path("datasets/crema_d_organized")
    
    if not source_dir.exists():
        print("CREMA-D AudioWAV directory not found!")
        return None
    
    # Emotion mapping for CREMA-D
    emotion_mapping = {
        'ANG': 'angry',
        'DIS': 'disgust', 
        'FEA': 'fearful',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad'
    }
    
    # Create target directories
    for emotion in emotion_mapping.values():
        (target_dir / emotion).mkdir(parents=True, exist_ok=True)
    
    # Process files
    file_count = 0
    emotion_counts = defaultdict(int)
    
    for audio_file in source_dir.glob("*.wav"):
        # Parse filename: 1091_IEO_SAD_HI.wav
        parts = audio_file.stem.split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_mapping:
                emotion = emotion_mapping[emotion_code]
                target_path = target_dir / emotion / audio_file.name
                shutil.copy2(audio_file, target_path)
                emotion_counts[emotion] += 1
                file_count += 1
    
    print(f"Organized {file_count} files from CREMA-D")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} files")
    
    return target_dir

def organize_emodb():
    """Organize EmoDB dataset by emotion."""
    print("\nOrganizing EmoDB dataset...")
    
    source_dir = Path("datasets/emodb/wav")
    target_dir = Path("datasets/emodb_organized")
    
    if not source_dir.exists():
        print("EmoDB wav directory not found!")
        return None
    
    # Emotion mapping for EmoDB (based on filename encoding)
    # Format: 08a01Fd.wav where 'F' indicates emotion
    emotion_mapping = {
        'W': 'happy',    # Wohlgefallen (pleasure)
        'L': 'sad',      # Trauer (sadness)
        'A': 'angry',    # Ärger (anger)
        'F': 'fearful',  # Furcht (fear)
        'E': 'disgust',  # Ekel (disgust)
        'T': 'surprised', # Überraschung (surprise)
        'N': 'neutral'   # Neutral
    }
    
    # Create target directories
    for emotion in emotion_mapping.values():
        (target_dir / emotion).mkdir(parents=True, exist_ok=True)
    
    # Process files
    file_count = 0
    emotion_counts = defaultdict(int)
    
    for audio_file in source_dir.glob("*.wav"):
        # Parse filename: 08a01Fd.wav
        filename = audio_file.stem
        if len(filename) >= 6:
            emotion_code = filename[5]  # 6th character indicates emotion
            if emotion_code in emotion_mapping:
                emotion = emotion_mapping[emotion_code]
                target_path = target_dir / emotion / audio_file.name
                shutil.copy2(audio_file, target_path)
                emotion_counts[emotion] += 1
                file_count += 1
            else:
                print(f"Unknown emotion code '{emotion_code}' in {audio_file.name}")
        else:
            print(f"Invalid filename format: {audio_file.name}")
    
    print(f"Organized {file_count} files from EmoDB")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} files")
    
    return target_dir

def create_train_val_split(dataset_dir, val_split=0.2):
    """Create train/validation split for a dataset."""
    print(f"\nCreating train/validation split for {dataset_dir.name}...")
    
    train_dir = dataset_dir.parent / f"{dataset_dir.name}_train_val"
    val_dir = dataset_dir.parent / f"{dataset_dir.name}_train_val"
    
    # Create train/val directories
    for split in ['train', 'val']:
        for emotion_dir in dataset_dir.iterdir():
            if emotion_dir.is_dir():
                (train_dir / split / emotion_dir.name).mkdir(parents=True, exist_ok=True)
    
    # Split files
    total_files = 0
    split_counts = defaultdict(lambda: defaultdict(int))
    
    for emotion_dir in dataset_dir.iterdir():
        if emotion_dir.is_dir():
            emotion = emotion_dir.name
            audio_files = list(emotion_dir.glob("*.wav"))
            random.shuffle(audio_files)
            
            # Calculate split point
            split_idx = int(len(audio_files) * (1 - val_split))
            train_files = audio_files[:split_idx]
            val_files = audio_files[split_idx:]
            
            # Copy train files
            for audio_file in train_files:
                target_path = train_dir / "train" / emotion / audio_file.name
                shutil.copy2(audio_file, target_path)
                split_counts["train"][emotion] += 1
                total_files += 1
            
            # Copy val files
            for audio_file in val_files:
                target_path = train_dir / "val" / emotion / audio_file.name
                shutil.copy2(audio_file, target_path)
                split_counts["val"][emotion] += 1
                total_files += 1
    
    print(f"Created train/validation split with {total_files} total files")
    for split in ['train', 'val']:
        print(f"  {split}:")
        for emotion, count in split_counts[split].items():
            print(f"    {emotion}: {count} files")
    
    return train_dir

def create_metadata(dataset_dir, dataset_name):
    """Create metadata file for the dataset."""
    metadata = {
        "dataset": dataset_name,
        "description": f"Organized {dataset_name} dataset for emotion detection",
        "emotions": [],
        "total_files": 0,
        "format": "wav",
        "sample_rate": 16000,
        "train_val_split": True
    }
    
    emotion_counts = {}
    total_files = 0
    
    for emotion_dir in dataset_dir.iterdir():
        if emotion_dir.is_dir():
            emotion = emotion_dir.name
            file_count = len(list(emotion_dir.glob("*.wav")))
            emotion_counts[emotion] = file_count
            total_files += file_count
            metadata["emotions"].append(emotion)
    
    metadata["total_files"] = total_files
    metadata["emotion_counts"] = emotion_counts
    
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata for {dataset_name}")
    return metadata

def main():
    """Main function to prepare datasets."""
    print("Speech Emotion Dataset Preparation")
    print("=" * 40)
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Organize CREMA-D
    crema_organized = organize_crema_d()
    if crema_organized:
        crema_train_val = create_train_val_split(crema_organized)
        create_metadata(crema_train_val, "CREMA-D")
    
    # Organize EmoDB
    emodb_organized = organize_emodb()
    if emodb_organized:
        emodb_train_val = create_train_val_split(emodb_organized)
        create_metadata(emodb_train_val, "EmoDB")
    
    print("\n" + "=" * 40)
    print("Dataset preparation complete!")
    
    if crema_organized:
        print(f"CREMA-D ready at: {crema_organized}")
        print(f"CREMA-D train/val at: {crema_train_val}")
    
    if emodb_organized:
        print(f"EmoDB ready at: {emodb_organized}")
        print(f"EmoDB train/val at: {emodb_train_val}")
    
    print("\nNext steps:")
    print("1. Use the organized datasets for Whisper fine-tuning")
    print("2. The train/val splits are ready for training")
    print("3. Each emotion has its own directory with audio files")

if __name__ == "__main__":
    main() 