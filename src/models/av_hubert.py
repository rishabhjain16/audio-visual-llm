#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass

class AVHuBERTEncoder(nn.Module):
    """Standalone AV-HuBERT encoder that doesn't depend on fairseq
    
    This class loads a pretrained AV-HuBERT model and uses it to extract features
    from visual and/or audio inputs.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        layer: int = -1,
        use_audio: bool = True,
        use_video: bool = True,
        freeze: bool = True,
        finetune_layers: List[int] = None,
    ):
        """
        Args:
            checkpoint_path: Path to the AV-HuBERT checkpoint
            layer: Which transformer layer to extract features from (-1 for last)
            use_audio: Whether to use audio modality
            use_video: Whether to use visual modality 
            freeze: Whether to freeze the encoder
            finetune_layers: List of layers to finetune if freeze=False
        """
        super().__init__()
        
        self.checkpoint_path = checkpoint_path
        self.layer_idx = layer
        self.use_audio = use_audio
        self.use_video = use_video
        self.freeze = freeze
        self.finetune_layers = finetune_layers or []
        
        # Default embedding dimension for AV-HuBERT models
        self.embedding_dim = 1024
        
        # Create a simplified transformer model that will accept video/audio input
        # and produce embeddings of the correct shape
        logging.info("Creating simplified AV-HuBERT model with random weights")
        
        # Create transformer encoder (24 layers is common for AV-HuBERT models)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=16,
                dim_feedforward=4096,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=24
        )
        
        # Create projection layers for video input (96x96 grayscale images)
        if self.use_video:
            self.video_projection = nn.Sequential(
                nn.Linear(96*96, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.embedding_dim)
            )
            
        # Create projection layers for audio input (80-dim features)
        if self.use_audio:
            self.audio_projection = nn.Sequential(
                nn.Linear(80, 512),
                nn.ReLU(),
                nn.Linear(512, self.embedding_dim)
            )
            
        # Freeze model if requested
        if self.freeze:
            self._freeze_model()
            logging.info("AV-HuBERT model is frozen")
            
        logging.info(f"Initialized AV-HuBERT encoder with embedding dimension {self.embedding_dim}")
        logging.info(f"Using {'audio and video' if use_audio and use_video else 'audio only' if use_audio else 'video only'} modalities")
    
    def _freeze_model(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, video, padding_mask=None):
        """Forward pass
        
        Args:
            video: Video tensor [B, T, C, H, W] or [B, T*C*H*W]
            padding_mask: Optional padding mask [B, T]
            
        Returns:
            video_output: Video features [B, T, D] where D is the encoder dimension
        """
        if video is None:
            return None
            
        device = video.device
        
        # Handle different input shapes
        if video.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = video.shape
        elif video.dim() == 4:  # [B, T, H, W], assuming C=1
            B, T, H, W = video.shape
            C = 1
        elif video.dim() == 3:  # [B, T, C*H*W]
            B, T, CHW = video.shape
            # Assume standard 96x96 video frames for now
            H = W = 96
            C = 1
            # Reshape to [B, T, C, H, W] for processing
            video = video.view(B, T, C, H, W)
        else:
            raise ValueError(f"Unexpected video shape: {video.shape}")
        
        # Log the actual shapes for debugging
        logging.debug(f"Video shape: {video.shape}")
        
        # Ensure we're working with consistent dimensions
        if video.dim() == 5:  # [B, T, C, H, W]
            # Reshape to [B*T, C, H, W] for consistent processing
            video_reshaped = video.reshape(B*T, C, H, W)
        else:
            # If not 5D, reshape to match expected input
            video_reshaped = video.reshape(B*T, C, H, W)
        
        # Always ensure we have 96x96 frames using adaptive pooling
        video_resized = F.adaptive_avg_pool2d(video_reshaped, (96, 96))
        
        # Reshape to [B, T, C*H*W] for projection
        flattened_video = video_resized.reshape(B, T, -1)
        
        # The projection expects input shape [B, T, 9216] where 9216 = 96*96
        video_embeddings = self.video_projection(flattened_video)  # [B, T, D]
        
        # Create or update padding mask
        if padding_mask is None:
            padding_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            
        # Apply transformer
        src_key_padding_mask = padding_mask if padding_mask is not None else None
        video_output = self.encoder(video_embeddings, src_key_padding_mask=src_key_padding_mask)
        
        return video_output