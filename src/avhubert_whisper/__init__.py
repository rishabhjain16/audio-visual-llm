"""AVHuBERT-Whisper module for audio-visual speech recognition

This module combines the AVHuBERT model for visual feature extraction
with the Whisper model for audio feature extraction, connecting both
to a language model for transcription.
"""

# Import models
from .models import AVHuBERTEncoder, AVHuBERTWhisperModel, WhisperEncoder

# Import trainer
from .trainer import AVHuBERTWhisperTrainer

__all__ = [
    'AVHuBERTEncoder',
    'AVHuBERTWhisperModel',
    'WhisperEncoder',
    'AVHuBERTWhisperTrainer'
] 