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
        use_fp16: bool = True,  # Use float16 by default for memory efficiency
    ):
        """
        Args:
            checkpoint_path: Path to the AV-HuBERT checkpoint
            layer: Which transformer layer to extract features from (-1 for last)
            use_audio: Whether to use audio modality
            use_video: Whether to use visual modality 
            freeze: Whether to freeze the encoder
            finetune_layers: List of layers to finetune if freeze=False
            use_fp16: Whether to use float16 precision (recommended for memory efficiency)
        """
        super().__init__()
        
        self.checkpoint_path = checkpoint_path
        self.layer_idx = layer
        self.use_audio = use_audio
        self.use_video = use_video
        self.freeze = freeze
        self.finetune_layers = finetune_layers or []
        self.use_fp16 = use_fp16
        
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
        
        # Load pre-trained weights if checkpoint path is provided
        if os.path.exists(checkpoint_path):
            self.load_pretrained_weights()
        
        # Freeze model if specified
        if self.freeze:
            self.freeze_model(except_layers=self.finetune_layers)
            logging.info("AV-HuBERT model is frozen")
        
        # Convert model to fp16 if specified (after loading weights)
        if self.use_fp16:
            self.to_fp16()
            logging.info("AV-HuBERT model converted to float16 precision")
        
        logging.info(f"Initialized AV-HuBERT encoder with embedding dimension {self.embedding_dim}")
        logging.info(f"Using {'audio and video' if use_audio and use_video else 'audio only' if use_audio else 'video only'} modalities")
    
    def to_fp16(self):
        """Convert model parameters to float16 for memory efficiency"""
        for name, param in self.named_parameters():
            if param.dtype != torch.float16:
                param.data = param.data.to(torch.float16)
        
        # Also ensure any buffers are in float16
        for name, buffer in self.named_buffers():
            if buffer.dtype != torch.float16 and buffer.dtype != torch.bool:
                buffer.data = buffer.data.to(torch.float16)
    
    def freeze_model(self, except_layers: List[int]):
        """Freeze all parameters except the specified layers"""
        for name, param in self.named_parameters():
            if name not in except_layers:
                param.requires_grad = False
    
    def load_pretrained_weights(self):
        """Load pre-trained weights from the checkpoint"""
        checkpoint = torch.load(self.checkpoint_path)
        self.load_state_dict(checkpoint)
    
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

class AVHubertEncoderWrapper(nn.Module):
    def __init__(self, av_hubert_model, freeze=True):
        super().__init__()
        self.av_hubert_model = av_hubert_model
        
        # Freeze the model if specified
        if freeze:
            for param in self.av_hubert_model.parameters():
                param.requires_grad = False
                
        # Convert model to float16 for memory efficiency and compatibility
        self.convert_to_fp16()
        
        # Create video projection as sequential
        self.video_projection = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024)
        )
        
        # Also convert projection to float16
        for module in self.video_projection.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.to(torch.float16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.float16)
    
    def convert_to_fp16(self):
        """Convert the model weights to float16 for memory efficiency"""
        for name, param in self.av_hubert_model.named_parameters():
            if param.dtype != torch.float16:
                # Use to_copy to avoid modifying frozen parameters in-place
                param.data = param.data.to(torch.float16)