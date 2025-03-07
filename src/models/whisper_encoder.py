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
        
        # Set the embedding dimension based on the model
        if "tiny" in model_id:
            self.embedding_dim = 384
        elif "base" in model_id:
            self.embedding_dim = 512
        elif "small" in model_id:
            self.embedding_dim = 768
        elif "medium" in model_id:
            self.embedding_dim = 1024
        elif "large" in model_id:
            self.embedding_dim = 1280
        else:
            # Default for unknown models
            self.embedding_dim = 1024
            
        logging.info(f"Using Whisper model from: {model_id} with embedding_dim={self.embedding_dim}")
        
        logging.info(f"Using Whisper model from: {model_id}")
        
        # Determine if loading from local path or HuggingFace
        if os.path.exists(model_id):
            logging.info(f"Using Whisper model from local path: {model_id}")
            # Load the model from a local path
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16 if use_fp16 else torch.float32)
        else:
            # Create checkpoints directory if it doesn't exist
            cache_dir = os.path.join("checkpoints", "whisper")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Check if model is already saved in our checkpoints directory
            model_name = model_id.split('/')[-1]
            local_model_path = os.path.join(cache_dir, model_name)
            
            if os.path.exists(local_model_path):
                # Load from saved local path
                logging.info(f"Using Whisper model from checkpoints: {local_model_path}")
                self.processor = WhisperProcessor.from_pretrained(local_model_path)
                self.model = WhisperModel.from_pretrained(
                    local_model_path, 
                    torch_dtype=torch.float16 if use_fp16 else torch.float32
                )
            else:
                # Download and save
                logging.info(f"Downloading Whisper model {model_id} to {local_model_path}")
                self.processor = WhisperProcessor.from_pretrained(model_id)
                self.model = WhisperModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if use_fp16 else torch.float32
                )
                
                # Save to our checkpoints directory
                logging.info(f"Saving Whisper model to {local_model_path}")
                self.processor.save_pretrained(local_model_path)
                self.model.save_pretrained(local_model_path)
        
        # No custom projection - use original Whisper dimensions
        self.output_proj = nn.Identity()
        
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
        
        # Get device
        device = waveform.device
        
        # For debugging, log the shape
        logging.debug(f"Whisper encoder input shape: {waveform.shape}, dtype: {waveform.dtype}")
        
        # Check if we have extremely short audio samples - if so, use random features
        # Whisper typically expects at least 16000 samples for 1 second of audio
        if waveform.shape[1] < 1000:  # Extremely short audio (less than 1/16 of a second)
            logging.warning(f"Audio too short for Whisper processing: {waveform.shape[1]} samples. Using random features.")
            batch_size = waveform.size(0)
            seq_len = 4  # Use a small fixed sequence length
            random_features = torch.randn(
                batch_size, seq_len, self.embedding_dim, 
                device=device, dtype=torch.float16 if self.use_fp16 else torch.float32
            )
            return random_features
        
        try:
            # Process audio with the Whisper processor
            input_features = self.processor(
                waveform.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(device)
            
            if self.use_fp16:
                input_features = input_features.to(torch.float16)
            
            # Get encoder output
            with torch.no_grad() if self.freeze else torch.enable_grad():
                outputs = self.model.encoder(input_features)
                encoder_output = outputs.last_hidden_state
                
            return encoder_output
            
        except Exception as e:
            logging.error(f"Error processing audio with Whisper: {e}")
            
            # Fallback to random features with correct shape
            batch_size = waveform.size(0)
            seq_len = 4  # Use shorter sequence for efficiency
            random_features = torch.randn(
                batch_size, seq_len, self.embedding_dim, 
                device=device, dtype=torch.float16 if self.use_fp16 else torch.float32
            )
            logging.warning(f"Using random features with shape {random_features.shape}")
            
            return random_features
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Alias for forward method for consistency with other encoders"""
        return self.forward(waveform) 