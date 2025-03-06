#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config

def check_file_exists(path, description):
    """Check if a file exists and print appropriate message"""
    if os.path.exists(path):
        print(f"✅ {description} found at: {path}")
        return True
    else:
        print(f"❌ {description} NOT FOUND at: {path}")
        return False

def check_directory_contents(path, description, expected_files=None):
    """Check directory contents and print summary"""
    if os.path.isdir(path):
        print(f"✅ {description} directory found at: {path}")
        files = list(os.listdir(path))
        print(f"   - Contains {len(files)} files/directories")
        
        # Print first few files as examples
        if files:
            print(f"   - Examples: {', '.join(files[:5])}")
            if len(files) > 5:
                print(f"   - ... and {len(files) - 5} more")
        
        # Check for expected files if provided
        if expected_files:
            missing = [f for f in expected_files if f not in files]
            if missing:
                print(f"❌ Missing expected files: {', '.join(missing)}")
            else:
                print(f"✅ All expected files are present")
                
        return True
    else:
        print(f"❌ {description} directory NOT FOUND at: {path}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check model paths for AVSR-LLM")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✅ Configuration loaded from: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    print("\n==== Checking AV-HuBERT Encoder Path ====")
    if hasattr(config.model, "av_encoder_path"):
        path = config.model.av_encoder_path
        check_file_exists(path, "AV-HuBERT checkpoint")
        # If it's a pickle or PT file, we expect a single file
    else:
        print("❌ av_encoder_path not specified in config")
    
    print("\n==== Checking Whisper Encoder Path ====")
    if hasattr(config.model, "whisper_path"):
        path = config.model.whisper_path
        # Check if it's a local path or a model ID
        if os.path.exists(path):
            # Local path
            check_directory_contents(path, "Whisper model", 
                                    ["config.json", "model.safetensors", "tokenizer.json"])
        else:
            print(f"ℹ️ Using Whisper model ID: {path}")
            print(f"   - Will be downloaded from Hugging Face if not cached")
    else:
        print("❌ whisper_path not specified in config")
    
    print("\n==== Checking LLM Path ====")
    if hasattr(config.model, "llm_path"):
        path = config.model.llm_path
        # Check if it's a local path or a model ID
        if os.path.exists(path):
            # Local path
            check_directory_contents(path, "LLM model", 
                                   ["config.json", "model.safetensors", "tokenizer.json"])
            
            # Check for model shards (common in larger models)
            shard_files = [f for f in os.listdir(path) 
                          if f.startswith("model-") and (f.endswith(".safetensors") or f.endswith(".bin"))]
            if shard_files:
                print(f"✅ Found {len(shard_files)} model shards: {', '.join(shard_files[:3])}")
                if len(shard_files) > 3:
                    print(f"   - ... and {len(shard_files) - 3} more")
        else:
            print(f"ℹ️ Using LLM model ID: {path}")
            print(f"   - Will be downloaded from Hugging Face if not cached")
    else:
        print("❌ llm_path not specified in config")
    
    print("\n==== Checking Data Path ====")
    if hasattr(config, "data") and hasattr(config.data, "path"):
        path = config.data.path
        if check_directory_contents(path, "Data"):
            # Check for manifest files
            manifest_path = os.path.join(path, getattr(config.data, "train_manifest", "train.tsv"))
            check_file_exists(manifest_path, "Training manifest")
            
            labels_path = os.path.join(path, getattr(config.data, "train_labels", "train.wrd"))
            check_file_exists(labels_path, "Training labels")
    else:
        print("❌ data.path not specified in config")
    
    print("\n==== Checking GPU Availability ====")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA is available with {device_count} device(s)")
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                free_mem_gb = free_mem / (1024 ** 3)
                total_mem_gb = total_mem / (1024 ** 3)
                print(f"   - GPU {i}: {props.name}, {props.total_memory / (1024**2):.0f} MB total, "
                     f"{free_mem_gb:.1f} GB free / {total_mem_gb:.1f} GB total")
        else:
            print("❌ CUDA is not available")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
    
    print("\nChecking complete. Fix any issues with paths before running training.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 