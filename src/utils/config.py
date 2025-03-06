#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sampling_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    fmax: int = 8000
    mel_scale: str = "htk"
    normalize: bool = True
    mean: float = 0.0
    std: float = 1.0


@dataclass
class VideoConfig:
    """Video processing configuration"""
    resize: List[int] = field(default_factory=lambda: [224, 224])
    crop_size: List[int] = field(default_factory=lambda: [224, 224])
    fps: int = 25
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    use_grayscale: bool = False
    face_detection_method: str = "dlib"  # options: dlib, mediapipe, mtcnn
    use_mouth_roi: bool = False
    roi_scale: float = 1.5  # Scale factor for ROI extraction


@dataclass
class ModelConfig:
    """Model configuration"""
    # Paths
    av_encoder_path: str = ""
    llm_path: str = ""
    whisper_path: str = "openai/whisper-small"
    
    # Architecture
    audio_encoder_type: str = "hubert"  # options: hubert, wav2vec2, whisper
    video_encoder_type: str = "resnet"  # options: resnet, efficientnet, avhubert
    fusion_type: str = "cross_attention"  # options: concat, cross_attention, multimodal_adapter
    use_audio: bool = True
    use_video: bool = True
    
    # Model dimensions
    audio_dim: int = 768
    video_dim: int = 512
    fusion_dim: int = 1024
    llm_dim: int = 4096
    
    # Model parameters
    dropout: float = 0.1
    adapter_dim: int = 256
    adapter_dropout: float = 0.1
    num_adapter_layers: int = 2
    
    # AV-HuBERT parameters
    avhubert_layer: int = -1
    finetune_avhubert_layers: List[int] = field(default_factory=list)
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Training parameters
    freeze_av_encoder: bool = True
    freeze_audio_encoder: bool = True
    freeze_llm: bool = True
    unfreeze_layer_norms: bool = True
    unfreeze_adapters: bool = True
    freeze_fusion: bool = False
    
    # Generation parameters
    max_length: int = 256
    num_beams: int = 5
    prompt_template: str = "Transcribe the speech: "


@dataclass
class DataConfig:
    """Data configuration"""
    path: str = ""
    train_file: str = "train.json"
    val_file: str = "val.json"
    test_file: str = "test.json"
    train_manifest: str = "train.tsv"
    train_labels: str = "train.wrd"
    val_manifest: str = "val.tsv"
    val_labels: str = "val.wrd"
    test_manifest: str = "test.tsv"
    test_labels: str = "test.wrd"
    max_video_length: int = 1000
    max_audio_length: int = 250000
    max_text_length: int = 256
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    num_epochs: int = 50
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    clip_grad_norm: float = 1.0
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    seed: int = 42
    fp16: bool = True
    scheduler_type: str = "linear"  # options: linear, cosine, constant
    gradient_accumulation_steps: int = 1
    optimize_memory: bool = True
    early_stopping_patience: int = 5
    max_grad_norm: float = 1.0
    resume_from_checkpoint: Optional[str] = None


@dataclass
class AVSRConfig:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


def dict_to_object(d):
    """Convert dictionary to object for dot notation access"""
    if isinstance(d, dict):
        class Config:
            pass
        obj = Config()
        for k, v in d.items():
            setattr(obj, k, dict_to_object(v))
        return obj
    elif isinstance(d, list):
        return [dict_to_object(x) for x in d]
    else:
        return d


def load_config(config_path):
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration object with dot notation access
    """
    logging.info(f"Loading configuration from {config_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Convert to object with dot notation
        config_obj = dict_to_object(config)
        
        return config_obj
        
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise


def save_config(config: AVSRConfig, output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration object
        output_path: Path to save configuration file
    """
    output_path = Path(output_path)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary
    config_dict = {}
    for section in ["model", "data", "training", "audio", "video"]:
        section_config = getattr(config, section)
        config_dict[section] = {}
        for key, value in section_config.__dict__.items():
            config_dict[section][key] = value
    
    # Save to YAML file
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False, indent=2)
    
    logging.info(f"Configuration saved to {output_path}")


def create_default_config() -> AVSRConfig:
    """
    Create default configuration
    
    Returns:
        Default configuration object
    """
    return AVSRConfig() 