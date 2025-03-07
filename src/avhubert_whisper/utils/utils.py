#!/usr/bin/env python3
"""
Utility functions for AVHuBERT-Whisper models.
"""

import os
import torch
import logging
import yaml
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    # Create formatters
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Logging to {log_file}")
    
    logging.info(f"Logging level set to {logging.getLevelName(level)}")
    
def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        raise

@dataclass
class AVHuBERTWhisperConfig:
    """Configuration for AVHuBERT-Whisper models"""
    
    # Model configuration
    llm_path: str = "meta-llama/Llama-2-7b-chat-hf"
    whisper_model: str = "openai/whisper-medium"
    avhubert_path: Optional[str] = None
    
    # Training configuration
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_epochs: int = 10
    batch_size: int = 4
    grad_accum_steps: int = 4
    max_grad_norm: float = 0.5
    warmup_ratio: float = 0.1
    
    # Data configuration
    data_path: str = "data"
    max_audio_length: int = 30  # seconds
    max_video_length: int = 300  # frames
    max_seq_len: int = 256
    
    # Model specific configuration
    modality: str = "both"  # "audio", "video", or "both"
    use_fp16: bool = False
    freeze_encoders: bool = True
    freeze_llm: bool = False
    fusion_scale: float = 0.5  # Weight for audio in fusion (0.5 = equal weight)
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Output configuration
    output_dir: str = "outputs/avhubert_whisper"
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def save(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = asdict(self)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logging.info(f"Saved configuration to {config_path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AVHuBERTWhisperConfig':
        """Create configuration from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AVHuBERTWhisperConfig':
        """Load configuration from YAML file"""
        config_dict = load_config(config_path)
        return cls.from_dict(config_dict) 