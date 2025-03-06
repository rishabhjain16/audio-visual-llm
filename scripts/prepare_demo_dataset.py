#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import sys
import random
import shutil
import argparse
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a demo dataset for AVSR-LLM testing")
    parser.add_argument("--source_data", type=str, required=True, 
                        help="Path to source dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for demo dataset")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to include in demo dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def create_demo_dataset(source_data, output_dir, num_samples=10, seed=42):
    """Create a small demo dataset from a larger dataset"""
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load source manifest
    train_manifest_path = os.path.join(source_data, "train.tsv")
    if not os.path.exists(train_manifest_path):
        raise FileNotFoundError(f"Train manifest not found: {train_manifest_path}")
    
    # Load source labels
    train_labels_path = os.path.join(source_data, "train.wrd")
    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(f"Train labels not found: {train_labels_path}")
    
    # Read manifest file
    with open(train_manifest_path, "r") as f:
        manifest_lines = f.readlines()
    
    # Read labels file
    with open(train_labels_path, "r") as f:
        labels_lines = f.readlines()
    
    # Create a map of sample IDs to labels
    labels_map = {}
    for line in labels_lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            sample_id, text = parts
            labels_map[sample_id] = text
    
    # Parse manifest
    samples = []
    for line in manifest_lines:
        parts = line.strip().split("	")
        if len(parts) >= 2:
            sample_id = parts[0]
            if sample_id in labels_map:
                audio_path = parts[1] if len(parts) >= 2 else "NA"
                video_path = parts[2] if len(parts) >= 3 else "NA"
                samples.append({
                    "id": sample_id,
                    "audio_path": audio_path,
                    "video_path": video_path,
                    "text": labels_map[sample_id]
                })
    
    # Select random samples
    selected_samples = random.sample(samples, min(num_samples, len(samples)))
    
    print(f"Selected {len(selected_samples)} samples for demo dataset")
    
    # Create train and val splits
    train_samples = selected_samples[:len(selected_samples)//2]
    val_samples = selected_samples[len(selected_samples)//2:]
    
    # Create output files for train set
    with open(os.path.join(output_dir, "train.tsv"), "w") as f_manifest:
        for sample in train_samples:
            # Write manifest line
            f_manifest.write(f"{sample['id']}	{sample['audio_path']}	{sample['video_path']}\n")
    
    with open(os.path.join(output_dir, "train.wrd"), "w") as f_labels:
        for sample in train_samples:
            # Write labels line
            f_labels.write(f"{sample['id']} {sample['text']}\n")
    
    # Create output files for val set
    with open(os.path.join(output_dir, "val.tsv"), "w") as f_manifest:
        for sample in val_samples:
            # Write manifest line
            f_manifest.write(f"{sample['id']}	{sample['audio_path']}	{sample['video_path']}\n")
    
    with open(os.path.join(output_dir, "val.wrd"), "w") as f_labels:
        for sample in val_samples:
            # Write labels line
            f_labels.write(f"{sample['id']} {sample['text']}\n")
    
    print(f"Created demo dataset with {len(train_samples)} train samples and {len(val_samples)} val samples")
    
    # Copy a README to explain the dataset
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"""# Demo Dataset for AVSR-LLM

This is a small demo dataset for testing the AVSR-LLM model. It contains {len(selected_samples)} samples from the original dataset.

- Train split: {len(train_samples)} samples
- Validation split: {len(val_samples)} samples

This dataset is meant for testing purposes only and should not be used for real model evaluation.
""")

def main():
    args = parse_args()
    
    # Create demo dataset
    create_demo_dataset(
        args.source_data,
        args.output_dir,
        args.num_samples,
        args.seed
    )

if __name__ == "__main__":
    main() 