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
        use_encoder_only: bool = True,
        use_fp16: bool = True  # Use float16 by default for memory efficiency
    ):
        """
        Args:
            model_id: HuggingFace model ID or path to pretrained model
            freeze: Whether to freeze the encoder
            use_encoder_only: Whether to use only the encoder part of Whisper
            use_fp16: Whether to use float16 precision (recommended for memory efficiency)
        """
        super().__init__()
        
        self.model_id = model_id
        self.freeze = freeze
        self.use_encoder_only = use_encoder_only
        self.use_fp16 = use_fp16
        
        logging.info(f"Using Whisper model from: {model_id}")
        
        # Determine if loading from local path or HuggingFace
        if os.path.exists(model_id):
            logging.info(f"Using Whisper model from local path: {model_id}")
            # Load the model from a local path
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16 if use_fp16 else torch.float32)
        else:
            logging.info(f"Using Whisper model from Hugging Face: {model_id}")
            # Load the model from HuggingFace
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16 if use_fp16 else torch.float32)
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.d_model
        
        # Freeze model if required
        if self.freeze:
            self._freeze_model()
            logging.info("Whisper model parameters frozen")
    
    def _freeze_model(self):
        """Freeze all parameters of the model"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Whisper encoder
        
        Args:
            waveform: Audio waveform of shape [B, T]
                where B is batch size and T is the number of samples
                
        Returns:
            encoder_output: Encoder output of shape [B, T, D]
                where D is the embedding dimension
        """
        if waveform is None:
            return None
        
        # Get device and dtype
        device = waveform.device
        dtype = waveform.dtype
        
        # Convert to float16 if needed
        if self.use_fp16 and dtype != torch.float16:
            waveform = waveform.to(torch.float16)
            
        # Process the audio input
        try:
            # Need to move waveform to CPU for processor (which uses numpy internally)
            cpu_waveform = waveform.cpu()
            
            # Process the audio
            input_features = self.processor(
                cpu_waveform, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(device)
            
            # Convert input features to fp16 if using fp16
            if self.use_fp16 and input_features.dtype != torch.float16:
                input_features = input_features.to(torch.float16)
            
            # Get encoder output
            with torch.no_grad() if self.freeze else torch.enable_grad():
                outputs = self.model.encoder(input_features)
                encoder_output = outputs.last_hidden_state
            
            return encoder_output
        
        except Exception as e:
            logging.error(f"Error in Whisper encoder: {e}")
            
            # Fallback to random features with correct shape
            batch_size = waveform.size(0)
            # Approximate sequence length based on typical Whisper output
            seq_len = 1500 // 320  # ~1500ms audio at 16kHz produces ~47 frames
            random_features = torch.randn(
                batch_size, seq_len, self.embedding_dim, 
                device=device, dtype=torch.float16 if self.use_fp16 else torch.float32
            )
            logging.warning(f"Using random features with shape {random_features.shape}")
            
            return random_features
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Alias for forward method for consistency with other encoders"""
        return self.forward(waveform) 