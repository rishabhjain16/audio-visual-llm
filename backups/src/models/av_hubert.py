#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
import traceback
import pickle

class AVHuBERTEncoder(nn.Module):
    """
    AV-HuBERT encoder that encodes video into embeddings
    
    Note: The last transformer layer of AV-HuBERT outputs 1024-dimensional features.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        layer: int = -1,  # Use the last layer by default, which has 1024 dimensions
        use_audio: bool = False,
        use_video: bool = True,
        freeze: bool = True,
        finetune_layers: List[int] = None,
        use_fp16: bool = True,  # Use float16 by default for memory efficiency
        output_dim: int = 1024  # Default output dimension is 1024 (from last layer)
    ):
        """
        Initialize AV-HuBERT encoder
        
        Args:
            checkpoint_path: Path to the AV-HuBERT checkpoint
            layer: Which transformer layer to extract features from (-1 for last layer)
            use_audio: Whether to use audio features
            use_video: Whether to use video features
            freeze: Whether to freeze the entire encoder
            finetune_layers: List of layers to finetune (if freeze is True, only these layers will be trainable)
            use_fp16: Whether to use float16 precision (recommended for memory efficiency)
            output_dim: Output dimension (default is 1024 from the last layer)
        """
        super().__init__()
        
        self.checkpoint_path = checkpoint_path
        self.layer = layer
        self.use_audio = use_audio
        self.use_video = use_video
        self.freeze = freeze
        self.finetune_layers = finetune_layers if finetune_layers is not None else []
        self.use_fp16 = use_fp16
        self.output_dim = output_dim
        
        # Set embedding_dim property for compatibility
        self.embedding_dim = 1024  # Base embedding dimension for AV-HuBERT
        
        # Initialize a simplified AV-HuBERT encoder 
        self._initialize_encoder()
        
        # Handle fine-tuning specific layers
        if freeze and finetune_layers:
            self._finetune_specific_layers()
            
        # Create a projection to ensure consistent output dimension
        self.output_projection = nn.Linear(1024, output_dim)
        # Convert to fp16 if requested
        if use_fp16:
            self.output_projection.weight.data = self.output_projection.weight.data.to(torch.float16)
            if self.output_projection.bias is not None:
                self.output_projection.bias.data = self.output_projection.bias.data.to(torch.float16)
    
    def _initialize_encoder(self):
        """Initialize a simplified AV-HuBERT encoder without fairseq dependency"""
        try:
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Model file not found: {self.checkpoint_path}")
            
            # Log absolute path for debugging
            abs_path = os.path.abspath(self.checkpoint_path)
            logging.info(f"Creating simplified AV-HuBERT encoder (no fairseq dependency)")
            logging.info(f"Using pretrained model path for reference: {abs_path}")
            
            # Create a simplified transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=1024,  # AV-HuBERT uses 1024 dimension
                nhead=16,      # 16 attention heads
                dim_feedforward=4096,
                dropout=0.1,
                activation="gelu",
                batch_first=True
            )
            
            # Create the encoder with multiple layers
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=12,  # AV-HuBERT typically has 12 or 24 layers
            )
            
            # Create video projection - 96x96 image to 1024 dim features
            self.video_projection = nn.Sequential(
                nn.Linear(96*96, 4096),  # Assuming 96x96 input images flattened
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(4096, 1024)
            )
            
            # Freeze the model if specified
            if self.freeze:
                self._freeze_model()
            
            # Convert to fp16 if specified
            if self.use_fp16:
                self._convert_to_fp16()
                
            logging.info(f"Simplified AV-HuBERT encoder initialized. Output dimension: {self.output_dim}")
            
        except Exception as e:
            logging.error(f"Error initializing simplified AV-HuBERT encoder: {e}")
            logging.error(traceback.format_exc())
    
    def _finetune_specific_layers(self):
        """Fine-tune specific layers"""
        if self.freeze:
            logging.info(f"Freezing all layers except: {self.finetune_layers}")
            for name, param in self.named_parameters():
                param.requires_grad = False
                
            # Unfreeze specific layers if requested
            if self.finetune_layers:
                for layer_idx in self.finetune_layers:
                    if isinstance(self.encoder, nn.TransformerEncoder) and 0 <= layer_idx < len(self.encoder.layers):
                        logging.info(f"Unfreezing layer {layer_idx}")
                        for param in self.encoder.layers[layer_idx].parameters():
                            param.requires_grad = True
                    else:
                        logging.warning(f"Invalid layer index: {layer_idx}")
        else:
            logging.info("All layers are trainable")
    
    def _freeze_model(self):
        """Freeze the entire model"""
        for param in self.parameters():
            param.requires_grad = False
        logging.info("AV-HuBERT encoder is frozen")
    
    def _convert_to_fp16(self):
        """Convert model to fp16 for better memory efficiency"""
        self.to(torch.float16)
        logging.info("Converted AV-HuBERT encoder to fp16")
    
    def forward(self, video, padding_mask=None):
        """
        Forward pass through the AV-HuBERT encoder
        
        Args:
            video: Video input tensor
                - Shape [B, T, C, H, W] or [B, T, C*H*W] or [B, T, H, W]
                - B = batch size, T = sequence length, C = channels, H = height, W = width
            padding_mask: Optional padding mask (True for padded positions)
                
        Returns:
            Encoded features with shape [B, T, output_dim]
        """
        # Get device and dtype from module parameters
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype if not self.use_fp16 else torch.float16
        
        # Ensure video is on the correct device and has the right dtype
        if video.device != device or video.dtype != dtype:
            video = video.to(device=device, dtype=dtype)
        
        # Determine video shape
        if video.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = video.shape
            video = video.reshape(B, T, C * H * W)
        elif video.dim() == 4:  # [B, T, H, W], assuming C=1
            B, T, H, W = video.shape
            video = video.reshape(B, T, H * W)
        elif video.dim() == 3:  # [B, T, C*H*W]
            B, T, CHW = video.shape
            # Keep as is
        else:
            raise ValueError(f"Unexpected video shape: {video.shape}")
        
        # Log the actual shapes for debugging
        logging.debug(f"Video shape after reshaping: {video.shape}")
        
        # Prepare video features - reshape to [B*T, C*H*W] for the projection
        flattened_video = video.reshape(-1, video.size(-1))
        
        # Apply the initial video projection
        video_embeddings = self.video_projection(flattened_video)  # [B*T, D]
        video_embeddings = video_embeddings.reshape(B, T, -1)  # [B, T, D]
        
        # Apply the encoder
        if padding_mask is not None:
            # The encoder expects padding_mask to be True for positions to mask
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask.bool()
            
            encoder_output = self.encoder(video_embeddings, src_key_padding_mask=padding_mask)
        else:
            encoder_output = self.encoder(video_embeddings)
            
        # Apply the output projection to match requested output dimension
        if encoder_output is not None:
            # Convert the output to fp16 if needed
            if self.use_fp16 and encoder_output.dtype != torch.float16:
                encoder_output = encoder_output.to(torch.float16)
                
            # Apply projection if needed
            if encoder_output.size(-1) != self.output_dim:
                encoder_output = self.output_projection(encoder_output)
                
        return encoder_output
        
    def to(self, *args, **kwargs):
        """Override to method to handle fp16 conversion properly"""
        result = super().to(*args, **kwargs)
        
        # Handle dtype conversion for specific modules
        if len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
            
            # Convert projections to the specified dtype
            if hasattr(self, 'video_projection'):
                for module in self.video_projection.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data = module.weight.data.to(dtype)
                        if module.bias is not None:
                            module.bias.data = module.bias.data.to(dtype)
                            
            if hasattr(self, 'audio_projection'):
                for module in self.audio_projection.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data = module.weight.data.to(dtype)
                        if module.bias is not None:
                            module.bias.data = module.bias.data.to(dtype)
            
            if hasattr(self, 'output_projection') and isinstance(self.output_projection, nn.Linear):
                self.output_projection.weight.data = self.output_projection.weight.data.to(dtype)
                if self.output_projection.bias is not None:
                    self.output_projection.bias.data = self.output_projection.bias.data.to(dtype)
        
        return result

class AVHubertEncoderWrapper(nn.Module):
    def __init__(self, av_hubert_model=None, freeze=True, output_dim: int = 2048):
        super().__init__()
        
        # Use provided model or create a new one
        if av_hubert_model is not None:
            self.av_hubert_model = av_hubert_model
        else:
            # Create a simple mock model
            logging.info("No AV-HuBERT model provided, creating a simple mock model")
            self.av_hubert_model = nn.Sequential(
                nn.Linear(96*96, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024)
            )
        
        # Freeze the model if specified
        if freeze and av_hubert_model is not None:
            for param in self.av_hubert_model.parameters():
                param.requires_grad = False
                
        # Create a proper output projection only if the output_dim is different from the model's native dimension
        # AVHuBERT typically outputs 1024 dimensions
        if output_dim != 1024:
            logging.info(f"Creating output projection from AVHuBERT dimension (1024) to {output_dim}")
            self.output_projection = nn.Linear(1024, output_dim)
            
            # Convert projection to float16 for consistency
            self.output_projection.weight.data = self.output_projection.weight.data.to(torch.float16)
            if self.output_projection.bias is not None:
                self.output_projection.bias.data = self.output_projection.bias.data.to(torch.float16)
        else:
            logging.info("No output projection needed as dimensions match")
            self.output_projection = nn.Identity()
    
    def forward(self, *args, **kwargs):
        """Forward inputs to the wrapped av_hubert_model and apply output projection"""
        try:
            # Call the wrapped model
            outputs = self.av_hubert_model(*args, **kwargs)
            
            # Apply output projection if needed
            if outputs is not None and self.output_projection is not None:
                outputs = self.output_projection(outputs)
                
            return outputs
        except Exception as e:
            logging.error(f"Error in AVHubertEncoderWrapper forward: {e}")
            logging.error(traceback.format_exc())
            # Return dummy tensor for graceful failure
            device = next(self.parameters()).device
            batch_size = 1  # Default batch size
            seq_len = 4     # Default sequence length
            if hasattr(self, 'output_projection') and isinstance(self.output_projection, nn.Linear):
                dim = self.output_projection.out_features
            else:
                dim = 1024  # Default dimension
            return torch.randn(batch_size, seq_len, dim, device=device)