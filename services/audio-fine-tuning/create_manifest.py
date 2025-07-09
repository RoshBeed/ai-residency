#!/usr/bin/env python3
"""
Create CSV manifest files for CREMA-D train and val splits.
Each row: audio_path,label
"""
import csv
from pathlib import Path

def create_manifest(split_dir, output_csv):
    rows = []
    for emotion_dir in Path(split_dir).iterdir():
        if emotion_dir.is_dir():
            label = emotion_dir.name
            for audio_file in emotion_dir.glob("*.wav"):
                rows.append({
                    'audio_path': str(audio_file.resolve()),
                    'label': label
                })
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['audio_path', 'label'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {output_csv}")

def main():
    base = Path("datasets/crema_d_organized_train_val")
    for split in ['train', 'val']:
        split_dir = base / split
        output_csv = f"cremad_{split}.csv"
        create_manifest(split_dir, output_csv)

if __name__ == "__main__":
    main() 