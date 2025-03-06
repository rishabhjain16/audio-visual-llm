#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import json
from tqdm import tqdm
import requests
from huggingface_hub import snapshot_download

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.setup import setup_logging


MODEL_REGISTRY = {
    "avhubert_base": {
        "description": "AV-HuBERT Base model pretrained on LRS3, 95M parameters",
        "hf_id": "kensho/avhubert-base-lrs3",
        "local_path": "avhubert_base.pt",
        "config": "avhubert_base_config.json"
    },
    "avhubert_large": {
        "description": "AV-HuBERT Large model pretrained on LRS3, 317M parameters",
        "hf_id": "kensho/avhubert-large-lrs3",
        "local_path": "avhubert_large.pt",
        "config": "avhubert_large_config.json"
    },
    "llama2_7b": {
        "description": "Llama-2 7B model from Meta",
        "hf_id": "meta-llama/Llama-2-7b-hf",
        "local_path": "llama2_7b",
        "config": None,
        "is_dir": True,
        "requires_auth": True
    },
    "llama3_8b": {
        "description": "Llama-3 8B model from Meta",
        "hf_id": "meta-llama/Meta-Llama-3-8B",
        "local_path": "llama3_8b",
        "config": None,
        "is_dir": True,
        "requires_auth": True
    },
    "mistral_7b": {
        "description": "Mistral 7B v0.1 model",
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "local_path": "mistral_7b",
        "config": None,
        "is_dir": True,
        "requires_auth": False
    },
    "tinyllama_1.1b": {
        "description": "TinyLlama 1.1B model (good for testing)",
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "local_path": "tinyllama_1.1b",
        "config": None,
        "is_dir": True,
        "requires_auth": False
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download pretrained models for AVSR-LLM")
    
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        help="Model to download or 'all' to download all models")
    
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save downloaded models")
    
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if model exists")
    
    return parser.parse_args()


def download_hf_model(model_info, output_dir, force=False):
    """Download a model from Hugging Face Hub"""
    model_name = model_info["hf_id"]
    local_path = Path(output_dir) / model_info["local_path"]
    
    # Check if model already exists
    if local_path.exists() and not force:
        if model_info.get("is_dir", False):
            # For directory models, check if config.json exists
            if (local_path / "config.json").exists():
                logging.info(f"Model {model_name} already exists at {local_path}")
                return True
        else:
            logging.info(f"Model {model_name} already exists at {local_path}")
            return True
    
    logging.info(f"Downloading {model_name} to {local_path}")
    
    try:
        if model_info.get("is_dir", False):
            # Create parent directory
            os.makedirs(local_path.parent, exist_ok=True)
            
            # Download using snapshot_download for directory models
            snapshot_download(
                repo_id=model_name,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
        else:
            # Create parent directory
            os.makedirs(local_path.parent, exist_ok=True)
            
            # For file models, use direct API download
            response = requests.get(
                f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin",
                stream=True
            )
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
            
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            
            # Download config if specified
            if model_info["config"]:
                config_path = Path(output_dir) / model_info["config"]
                config_response = requests.get(
                    f"https://huggingface.co/{model_name}/resolve/main/config.json"
                )
                config_response.raise_for_status()
                with open(config_path, "wb") as f:
                    f.write(config_response.content)
        
        logging.info(f"Successfully downloaded {model_name}")
        return True
    
    except Exception as e:
        logging.error(f"Error downloading {model_name}: {e}")
        if model_info.get("requires_auth", False):
            logging.error("This model requires Hugging Face authentication.")
            logging.error("Please run 'huggingface-cli login' to authenticate.")
            logging.error("For gated models like Llama, ensure you have access permission.")
        return False


def main():
    args = parse_args()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(Path(args.output_dir) / "download_models.log")
    
    # Determine which models to download
    if args.model_name == "all":
        models_to_download = MODEL_REGISTRY.items()
    else:
        models_to_download = [(args.model_name, MODEL_REGISTRY[args.model_name])]
    
    # Download models
    success_count = 0
    for model_name, model_info in models_to_download:
        logging.info(f"Processing {model_name}: {model_info['description']}")
        
        if download_hf_model(model_info, args.output_dir, args.force):
            success_count += 1
    
    # Log summary
    total_models = len(models_to_download)
    logging.info(f"Download summary: {success_count}/{total_models} models successfully downloaded")
    
    if success_count < total_models:
        logging.warning("Not all models were downloaded successfully. Check the log for details.")
        sys.exit(1)
    else:
        logging.info("All models successfully downloaded.")
        sys.exit(0)


if __name__ == "__main__":
    main() 