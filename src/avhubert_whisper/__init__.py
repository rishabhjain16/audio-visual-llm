"""AVHuBERT-Whisper module for audio-visual speech recognition

This module combines the AVHuBERT model for visual feature extraction
with the Whisper model for audio feature extraction, connecting both
to a language model for transcription.
"""

# Import models
from .models import AVHuBERTEncoder, AVHuBERTWhisperModel, WhisperEncoder

# Import trainer
from .trainer import AVHuBERTWhisperTrainer

# Import utils
from .utils import (
    set_seed,
    setup_logging,
    load_config,
    AVHuBERTWhisperConfig
)

__all__ = [
    # Models
    'AVHuBERTEncoder',
    'AVHuBERTWhisperModel',
    'WhisperEncoder',
    
    # Trainer
    'AVHuBERTWhisperTrainer',
    
    # Utils
    'set_seed',
    'setup_logging',
    'load_config',
    'AVHuBERTWhisperConfig'
] 