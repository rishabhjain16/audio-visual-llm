#!/usr/bin/env python3
import os
import logging
import argparse
import torch
from transformers import WhisperProcessor, CLIPProcessor, AutoTokenizer
import yaml
import sys
from pathlib import Path
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from src.models.simple_avsr import SimpleAVSRModel
from src.data.simple_dataset import create_dataloaders
from src.trainer.simple_trainer import SimpleTrainer
from src.utils.setup import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Train AVSR-LLM model")
    parser.add_argument("--config", type=str, default="configs/simple.yaml", 
                        help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--llm_path", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Path or name of the LLM model")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-medium",
                        help="Path or name of the Whisper model")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                        help="Path or name of the CLIP model")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training (lower values like 2-4 improve stability)")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (more logging)")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments and set stable defaults
    config["training"] = config.get("training", {})
    config["model"] = config.get("model", {})
    config["data"] = config.get("data", {})
    
    # Data settings
    if args.data_path:
        config["data"]["path"] = args.data_path
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
        logging.info(f"Using batch size: {args.batch_size} (set via command line)")
    else:
        config["data"]["batch_size"] = config["data"].get("batch_size", 2)  # Default to 2
        logging.info(f"Using batch size: {config['data']['batch_size']} (from config)")
    
    # Model settings
    if args.llm_path:
        config["model"]["llm_path"] = args.llm_path
    if args.whisper_model:
        config["model"]["whisper_model"] = args.whisper_model
    if args.clip_model:
        config["model"]["clip_model"] = args.clip_model
    config["model"]["fp16"] = args.fp16 if args.fp16 else False  # Default to False
    
    # Training settings with stable defaults
    config["training"]["max_epochs"] = args.max_epochs if args.max_epochs else config["training"].get("max_epochs", 10)
    config["training"]["learning_rate"] = args.learning_rate if args.learning_rate else config["training"].get("learning_rate", 1e-5)
    config["training"]["weight_decay"] = config["training"].get("weight_decay", 0.01)
    config["training"]["grad_accum_steps"] = config["training"].get("grad_accum_steps", 4)
    config["training"]["max_grad_norm"] = config["training"].get("max_grad_norm", 0.5)
    config["training"]["warmup_ratio"] = config["training"].get("warmup_ratio", 0.1)
    
    # Set up output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(os.path.join(output_dir, "train.log"), level=log_level)
    
    # Set GPU device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    
    # Log configuration
    logging.info("=== Training Configuration ===")
    logging.info(f"Data path: {config['data']['path']}")
    logging.info(f"Batch size: {config['data']['batch_size']}")
    logging.info(f"Model: llm={config['model']['llm_path']}, whisper={config['model']['whisper_model']}, clip={config['model']['clip_model']}")
    logging.info(f"Training: epochs={config['training']['max_epochs']}, lr={config['training']['learning_rate']}")
    logging.info(f"Stability: grad_accum={config['training']['grad_accum_steps']}, max_grad_norm={config['training']['max_grad_norm']}")
    logging.info(f"Device: {device}, FP16: {config['model']['fp16']}")
    
    try:
        # Load processors
        logging.info("Loading processors...")
        whisper_processor = WhisperProcessor.from_pretrained(config["model"]["whisper_model"])
        clip_processor = CLIPProcessor.from_pretrained(config["model"]["clip_model"])
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["llm_path"])
        
        # Create dataloaders with error handling
        logging.info("Creating dataloaders...")
        train_dataloader = create_dataloaders(
            manifest_path=os.path.join(config["data"]["path"], config["data"]["train_manifest"]),
            label_path=os.path.join(config["data"]["path"], config["data"]["train_labels"]),
            root_dir=config["data"]["path"],
            whisper_processor=whisper_processor,
            clip_processor=clip_processor,
            tokenizer=tokenizer,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"].get("num_workers", 2),  # Reduced from default
            max_audio_length=config["data"]["max_audio_length"],
            max_video_length=config["data"]["max_video_length"],
            split="train",
        )
        
        val_dataloader = None
        if "val_manifest" in config["data"] and "val_labels" in config["data"]:
            val_dataloader = create_dataloaders(
                manifest_path=os.path.join(config["data"]["path"], config["data"]["val_manifest"]),
                label_path=os.path.join(config["data"]["path"], config["data"]["val_labels"]),
                root_dir=config["data"]["path"],
                whisper_processor=whisper_processor,
                clip_processor=clip_processor,
                tokenizer=tokenizer,
                batch_size=config["data"]["batch_size"],
                num_workers=config["data"].get("num_workers", 2),
                max_audio_length=config["data"]["max_audio_length"],
                max_video_length=config["data"]["max_video_length"],
                split="val",
            )
        
        # Create model with error handling
        logging.info("Creating model...")
        model = SimpleAVSRModel(
            llm_path=config["model"]["llm_path"],
            whisper_model=config["model"]["whisper_model"],
            clip_model=config["model"]["clip_model"],
            device=device,
            use_fp16=config["model"]["fp16"],
            modality="both",  # Use both modalities by default
            max_seq_len=config["data"].get("max_seq_len", 256),
        )
        
        # Create trainer with stable settings
        logging.info("Creating trainer...")
        trainer = SimpleTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            max_epochs=config["training"]["max_epochs"],
            output_dir=output_dir,
            device=device,
            fp16=config["model"]["fp16"],
            grad_accum_steps=config["training"]["grad_accum_steps"],
            log_interval=config["training"].get("log_interval", 10),
            save_interval=config["training"].get("save_interval", 1),
            max_grad_norm=config["training"]["max_grad_norm"],
            warmup_ratio=config["training"]["warmup_ratio"],
        )
        
        # Train model with error handling
        logging.info("Starting training...")
        try:
            results = trainer.train()
            logging.info(f"Training completed with results: {results}")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            logging.error(traceback.format_exc())
            return 1
        
        return 0
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 