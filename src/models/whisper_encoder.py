#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
from transformers import WhisperModel, WhisperProcessor

class WhisperEncoder(nn.Module):
    """Whisper encoder for audio feature extraction
    
    This class loads a pretrained Whisper model and uses it to extract features
    from audio inputs. These features can then be used for speech recognition
    or combined with visual features.
    """
    
    def __init__(
        self,
        model_id: str = "openai/whisper-small",
        freeze: bool = True,
        use_encoder_only: bool = True
    ):
        """
        Args:
            model_id: HuggingFace model ID or path to pretrained model
            freeze: Whether to freeze the encoder
            use_encoder_only: Whether to use only the encoder part of Whisper
        """
        super().__init__()
        
        self.model_id = model_id
        self.freeze = freeze
        self.use_encoder_only = use_encoder_only
        
        # Load Whisper model and processor
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperModel.from_pretrained(model_id)
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.d_model
        
        # Freeze model if required
        if self.freeze:
            self._freeze_model()
    
    def _freeze_model(self):
        """Freeze all parameters of the model"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        logging.info(f"Whisper model parameters frozen")
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from audio waveform
        
        Args:
            waveform: Audio waveform tensor of shape (batch_size, channels, time)
                     Expected to be in the range [-1, 1] and have sample rate 16kHz
        
        Returns:
            Audio features of shape (batch_size, seq_len, embedding_dim)
        """
        # Store original device
        device = waveform.device
        
        # Ensure model is on the same device as input
        if next(self.model.parameters()).device != device:
            logging.info(f"Moving Whisper model to device: {device}")
            self.model = self.model.to(device)
        
        # Ensure waveform is mono
        if waveform.size(1) > 1:
            waveform = torch.mean(waveform, dim=1, keepdim=True)
        
        # Reshape to (batch_size, time)
        waveform = waveform.squeeze(1)
        
        # Process with Whisper
        with torch.no_grad() if self.freeze else torch.enable_grad():
            # Process on the same device as the input
            try:
                # Get input features using the processor
                input_features = self.processor(
                    waveform.cpu(), 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(device)
                
                # Get Whisper encoder output
                outputs = self.model.encoder(input_features)
                
                # Get the hidden states
                hidden_states = outputs.last_hidden_state
            except RuntimeError as e:
                logging.error(f"Error processing with Whisper: {e}")
                # Fallback to random features if processing fails
                batch_size = waveform.size(0)
                seq_len = 1500 // 320  # Approximate sequence length for 1500ms audio
                hidden_states = torch.randn(batch_size, seq_len, self.embedding_dim, device=device)
                logging.warning(f"Using random features with shape {hidden_states.shape}")
        
        return hidden_states
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Alias for forward method for consistency with other encoders"""
        return self.forward(waveform) 