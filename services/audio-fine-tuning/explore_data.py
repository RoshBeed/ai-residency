#!/usr/bin/env python3
"""
Explore and analyze the speech emotion datasets.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import librosa
import librosa.display
from collections import defaultdict
import seaborn as sns

def analyze_dataset_structure():
    """Analyze the structure of our datasets."""
    print("Dataset Structure Analysis")
    print("=" * 50)
    
    datasets = {
        "CREMA-D": "datasets/crema_d_organized_train_val",
        "EmoDB": "datasets/emodb_organized_train_val"
    }
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{dataset_name}:")
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            print(f"  Dataset not found: {dataset_path}")
            continue
            
        # Analyze train/val structure
        for split in ['train', 'val']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                print(f"  {split.upper()}:")
                emotion_counts = {}
                total_files = 0
                
                for emotion_dir in split_dir.iterdir():
                    if emotion_dir.is_dir():
                        file_count = len(list(emotion_dir.glob("*.wav")))
                        emotion_counts[emotion_dir.name] = file_count
                        total_files += file_count
                        print(f"    {emotion_dir.name}: {file_count} files")
                
                print(f"    Total: {total_files} files")
        
        # Check metadata
        metadata_file = dataset_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"  Metadata: {metadata.get('total_files', 'N/A')} total files")
            print(f"  Emotions: {', '.join(metadata.get('emotions', []))}")

def analyze_audio_characteristics():
    """Analyze audio characteristics of the datasets."""
    print("\n\nAudio Characteristics Analysis")
    print("=" * 50)
    
    datasets = {
        "CREMA-D": "datasets/crema_d_organized_train_val/train",
        "EmoDB": "datasets/emodb_organized_train_val/train"
    }
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{dataset_name}:")
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            print(f"  Dataset not found: {dataset_path}")
            continue
        
        # Sample a few files from each emotion for analysis
        durations = []
        sample_rates = []
        emotion_samples = {}
        
        for emotion_dir in dataset_dir.iterdir():
            if emotion_dir.is_dir():
                emotion = emotion_dir.name
                audio_files = list(emotion_dir.glob("*.wav"))[:5]  # Sample 5 files per emotion
                
                emotion_samples[emotion] = []
                
                for audio_file in audio_files:
                    try:
                        # Load audio file
                        y, sr = librosa.load(audio_file, sr=None)
                        duration = librosa.get_duration(y=y, sr=sr)
                        
                        durations.append(duration)
                        sample_rates.append(sr)
                        emotion_samples[emotion].append({
                            'file': audio_file.name,
                            'duration': duration,
                            'sample_rate': sr,
                            'samples': len(y)
                        })
                        
                    except Exception as e:
                        print(f"    Error loading {audio_file}: {e}")
        
        # Print statistics
        if durations:
            print(f"  Duration stats:")
            print(f"    Mean: {np.mean(durations):.2f}s")
            print(f"    Min: {np.min(durations):.2f}s")
            print(f"    Max: {np.max(durations):.2f}s")
            print(f"    Std: {np.std(durations):.2f}s")
        
        if sample_rates:
            unique_sr = set(sample_rates)
            print(f"  Sample rates: {unique_sr}")
        
        # Print emotion-specific samples
        for emotion, samples in emotion_samples.items():
            if samples:
                avg_duration = np.mean([s['duration'] for s in samples])
                print(f"  {emotion}: avg duration = {avg_duration:.2f}s")

def create_visualizations():
    """Create visualizations of the datasets."""
    print("\n\nCreating Visualizations")
    print("=" * 50)
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    datasets = {
        "CREMA-D": "datasets/crema_d_organized_train_val",
        "EmoDB": "datasets/emodb_organized_train_val"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Speech Emotion Dataset Analysis', fontsize=16)
    
    for idx, (dataset_name, dataset_path) in enumerate(datasets.items()):
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            continue
        
        # Collect data for visualization
        train_counts = {}
        val_counts = {}
        
        for split in ['train', 'val']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                for emotion_dir in split_dir.iterdir():
                    if emotion_dir.is_dir():
                        file_count = len(list(emotion_dir.glob("*.wav")))
                        if split == 'train':
                            train_counts[emotion_dir.name] = file_count
                        else:
                            val_counts[emotion_dir.name] = file_count
        
        # Plot 1: Train/Val split comparison
        ax1 = axes[idx, 0]
        emotions = list(train_counts.keys())
        train_values = [train_counts.get(emotion, 0) for emotion in emotions]
        val_values = [val_counts.get(emotion, 0) for emotion in emotions]
        
        x = np.arange(len(emotions))
        width = 0.35
        
        ax1.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
        ax1.bar(x + width/2, val_values, width, label='Validation', alpha=0.8)
        
        ax1.set_xlabel('Emotion')
        ax1.set_ylabel('Number of Files')
        ax1.set_title(f'{dataset_name} - Train/Validation Split')
        ax1.set_xticks(x)
        ax1.set_xticklabels(emotions, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pie chart of emotion distribution
        ax2 = axes[idx, 1]
        total_train = sum(train_counts.values())
        if total_train > 0:
            sizes = list(train_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=emotions, autopct='%1.1f%%', 
                                               colors=colors, startangle=90)
            ax2.set_title(f'{dataset_name} - Emotion Distribution (Train)')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("  Saved visualization to: dataset_analysis.png")
    
    # Create additional analysis
    create_audio_analysis_plots()

def create_audio_analysis_plots():
    """Create detailed audio analysis plots."""
    print("  Creating audio analysis plots...")
    
    # Analyze a sample file from each emotion
    sample_files = {}
    dataset_path = "datasets/crema_d_organized_train_val/train"
    
    for emotion_dir in Path(dataset_path).iterdir():
        if emotion_dir.is_dir():
            audio_files = list(emotion_dir.glob("*.wav"))
            if audio_files:
                sample_files[emotion_dir.name] = audio_files[0]
    
    if not sample_files:
        return
    
    # Create spectrogram comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Audio Spectrograms by Emotion (CREMA-D)', fontsize=16)
    
    emotions = list(sample_files.keys())
    
    for idx, emotion in enumerate(emotions[:6]):  # Limit to 6 emotions
        audio_file = sample_files[emotion]
        row = idx // 3
        col = idx % 3
        
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=16000)
            
            # Create spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            # Plot
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', 
                                         ax=axes[row, col])
            axes[row, col].set_title(f'{emotion.capitalize()}')
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].set_ylabel('Frequency (Hz)')
            
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'Error loading\n{emotion}', 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f'{emotion.capitalize()}')
    
    plt.tight_layout()
    plt.savefig('audio_spectrograms.png', dpi=300, bbox_inches='tight')
    print("  Saved spectrograms to: audio_spectrograms.png")

def print_and_plot_examples():
    """Print example file names, their emotion label, and plot mel spectrograms."""
    print("\n\nExample Audio Files and Mel Spectrograms")
    print("=" * 50)
    
    dataset_path = "datasets/crema_d_organized_train_val/train"
    n_examples = 2  # Number of examples per emotion
    example_files = []
    
    for emotion_dir in Path(dataset_path).iterdir():
        if emotion_dir.is_dir():
            audio_files = list(emotion_dir.glob("*.wav"))[:n_examples]
            for audio_file in audio_files:
                example_files.append((audio_file, emotion_dir.name))
    
    # Print file names and labels
    for audio_file, label in example_files:
        print(f"File: {audio_file.name} | Label: {label}")
    
    # Plot mel spectrograms
    n = len(example_files)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    fig.suptitle('Example Mel Spectrograms (CREMA-D)', fontsize=16)
    
    for idx, (audio_file, label) in enumerate(example_files):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]
        try:
            y, sr = librosa.load(audio_file, sr=16000)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
            ax.set_title(f"{label.capitalize()}\n{audio_file.name}")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Mel Frequency')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\n{audio_file.name}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{label.capitalize()}\n{audio_file.name}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('example_melspectrograms.png', dpi=300, bbox_inches='tight')
    print("  Saved example mel spectrograms to: example_melspectrograms.png")

def print_dataset_summary():
    """Print a comprehensive summary of the datasets."""
    print("\n\nDataset Summary")
    print("=" * 50)
    
    summary = {
        "CREMA-D": {
            "path": "datasets/crema_d_organized_train_val",
            "description": "Crowd-sourced Emotional Multimodal Actors Dataset",
            "emotions": ["angry", "disgust", "fearful", "happy", "neutral", "sad"],
            "total_files": 7442,
            "format": "WAV",
            "sample_rate": "16kHz"
        },
        "EmoDB": {
            "path": "datasets/emodb_organized_train_val", 
            "description": "Berlin Database of Emotional Speech",
            "emotions": ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
            "total_files": 535,
            "format": "WAV",
            "sample_rate": "16kHz"
        }
    }
    
    for dataset_name, info in summary.items():
        print(f"\n{dataset_name}:")
        print(f"  Description: {info['description']}")
        print(f"  Emotions: {', '.join(info['emotions'])}")
        print(f"  Total Files: {info['total_files']:,}")
        print(f"  Format: {info['format']}")
        print(f"  Sample Rate: {info['sample_rate']}")
        print(f"  Path: {info['path']}")

def main():
    """Main function to explore the datasets."""
    print("Speech Emotion Dataset Exploration")
    print("=" * 60)
    
    # Analyze dataset structure
    analyze_dataset_structure()
    
    # Analyze audio characteristics
    analyze_audio_characteristics()
    
    # Create visualizations
    create_visualizations()
    
    # Print and plot examples
    print_and_plot_examples()

    # Print summary
    print_dataset_summary()
    
    print("\n" + "=" * 60)
    print("Exploration complete!")
    print("Check the generated PNG files for visualizations.")
    print("\nNext steps:")
    print("1. Review the dataset statistics above")
    print("2. Examine the generated visualizations")
    print("3. Proceed with Whisper fine-tuning preparation")

if __name__ == "__main__":
    main() 