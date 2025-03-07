#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
import traceback


class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion module for combining audio and video features
    
    This module uses cross-attention to fuse audio and video features.
    """
    
    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        fusion_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2
    ):
        """
        Args:
            audio_dim: Dimension of audio features
            video_dim: Dimension of video features
            fusion_dim: Dimension of fused features
            num_heads: Number of attention heads
            dropout: Dropout rate
            num_layers: Number of fusion layers
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.fusion_dim = fusion_dim
        
        logging.info(f"Initializing CrossModalFusion with audio_dim={audio_dim}, video_dim={video_dim}, fusion_dim={fusion_dim}")
        
        # Project audio and video to fusion dimension
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.video_proj = nn.Linear(video_dim, fusion_dim)
        
        # Layer normalization
        self.norm_audio = nn.LayerNorm(fusion_dim)
        self.norm_video = nn.LayerNorm(fusion_dim)
        
        # Cross-attention layers
        self.fusion_layers = nn.ModuleList([
            CrossAttentionLayer(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)
        self.output_norm = nn.LayerNorm(fusion_dim)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse audio and video features
        
        Args:
            audio_features: Audio features of shape (batch_size, seq_len, audio_dim)
            video_features: Video features of shape (batch_size, seq_len, video_dim)
            padding_mask: Padding mask where True indicates padding
            
        Returns:
            Fused features of shape (batch_size, seq_len, fusion_dim)
        """
        # Ensure sequences have the same length
        min_length = min(audio_features.size(1), video_features.size(1))
        audio_features = audio_features[:, :min_length]
        video_features = video_features[:, :min_length]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :min_length]
        
        # Check for dimension mismatches and handle them
        if audio_features.size(2) != self.audio_dim:
            logging.warning(f"Audio feature dimension {audio_features.size(2)} doesn't match expected {self.audio_dim}. Creating a new projection.")
            device = audio_features.device
            self.audio_proj = nn.Linear(audio_features.size(2), self.fusion_dim).to(device)
        
        if video_features.size(2) != self.video_dim:
            logging.warning(f"Video feature dimension {video_features.size(2)} doesn't match expected {self.video_dim}. Creating a new projection.")
            device = video_features.device
            self.video_proj = nn.Linear(video_features.size(2), self.fusion_dim).to(device)
        
        # Project features to fusion dimension
        audio_proj = self.audio_proj(audio_features)
        video_proj = self.video_proj(video_features)
        
        # Normalize
        audio_proj = self.norm_audio(audio_proj)
        video_proj = self.norm_video(video_proj)
        
        # Initialize with audio features
        fused = audio_proj
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            fused = layer(
                query=fused,
                key=video_proj,
                value=video_proj,
                padding_mask=padding_mask
            )
        
        # Final projection
        fused = self.output_proj(fused)
        fused = self.output_norm(fused)
        
        return fused


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for fusion"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Ensure embed_dim is divisible by num_heads to avoid attention issues
        if embed_dim % num_heads != 0:
            old_num_heads = num_heads
            num_heads = max(1, embed_dim // 64)  # Use 64-dim per head as default
            logging.warning(f"Adjusted num_heads from {old_num_heads} to {num_heads} to be compatible with embed_dim={embed_dim}")
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            padding_mask: Padding mask where True indicates padding
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention block
        residual = query
        
        # Convert padding_mask to attention mask if provided
        attn_mask = None
        if padding_mask is not None:
            # Don't use attn_mask with MultiheadAttention, it causes shape issues
            # Just use key_padding_mask which is properly handled internally
            pass
        
        # Apply cross-attention
        attn_output, _ = self.self_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=padding_mask,
            attn_mask=None  # Set to None to avoid shape issues
        )
        
        # Add & norm
        attn_output = self.dropout(attn_output)
        out = self.norm1(residual + attn_output)
        
        # Feed-forward block
        residual = out
        ffn_output = self.ffn(out)
        ffn_output = self.dropout(ffn_output)
        out = self.norm2(residual + ffn_output)
        
        return out


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion that joins audio and video representations
    and projects them to the expected output dimension.
    """
    def __init__(self, audio_dim, video_dim, output_dim):
        super().__init__()
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.output_dim = output_dim
        
        # Create projection layers for each modality
        self.audio_proj = nn.Linear(audio_dim, output_dim)
        self.video_proj = nn.Linear(video_dim, output_dim)
        
        # Create fusion projection (audio+video concatenated)
        concat_dim = audio_dim + video_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, audio_features, video_features):
        """
        Fuse audio and video features through concatenation and projection
        
        Args:
            audio_features: Audio features [B, T, D_audio]
            video_features: Video features [B, T, D_video]
            
        Returns:
            Fused features [B, T, output_dim]
        """
        # Handle case where both inputs are None
        if audio_features is None and video_features is None:
            raise ValueError("Both audio and video features are None, cannot perform fusion")
        
        # Log detailed information about input features
        if audio_features is not None:
            logging.info(f"Audio features: shape={audio_features.shape}, dtype={audio_features.dtype}, device={audio_features.device}")
        else:
            logging.info("Audio features: None")
            
        if video_features is not None:
            logging.info(f"Video features: shape={video_features.shape}, dtype={video_features.dtype}, device={video_features.device}")
        else:
            logging.info("Video features: None")
        
        # Single modality case - handle it first to avoid unnecessary operations
        if audio_features is None:
            # Only video available
            logging.info(f"Using video-only projection to {self.output_dim}")
            # Ensure video features are float32
            if video_features.dtype != torch.float32:
                video_features = video_features.to(torch.float32)
                
            # Check if video dimensions match expected dimensions
            actual_video_dim = video_features.size(-1)
            if actual_video_dim != self.video_dim:
                logging.warning(f"Video feature dimension mismatch: {actual_video_dim} vs expected {self.video_dim}")
                logging.info(f"Recreating video projection with correct dimensions: {actual_video_dim} -> {self.output_dim}")
                self.video_dim = actual_video_dim
                self.video_proj = nn.Linear(actual_video_dim, self.output_dim).to(device=video_features.device, dtype=torch.float32)
                
            # Ensure projection weights are float32
            if self.video_proj.weight.dtype != torch.float32:
                self.video_proj = self.video_proj.to(torch.float32)
                
            return self.video_proj(video_features)
            
        if video_features is None:
            # Only audio available
            logging.info(f"Using audio-only projection to {self.output_dim}")
            # Ensure audio features are float32
            if audio_features.dtype != torch.float32:
                audio_features = audio_features.to(torch.float32)
                
            # Check if audio dimensions match expected dimensions
            actual_audio_dim = audio_features.size(-1)
            if actual_audio_dim != self.audio_dim:
                logging.warning(f"Audio feature dimension mismatch: {actual_audio_dim} vs expected {self.audio_dim}")
                logging.info(f"Recreating audio projection with correct dimensions: {actual_audio_dim} -> {self.output_dim}")
                self.audio_dim = actual_audio_dim
                self.audio_proj = nn.Linear(actual_audio_dim, self.output_dim).to(device=audio_features.device, dtype=torch.float32)
                
            # Ensure projection weights are float32
            if self.audio_proj.weight.dtype != torch.float32:
                self.audio_proj = self.audio_proj.to(torch.float32)
                
            return self.audio_proj(audio_features)
        
        # Ensure both features are float32 for better compatibility
        if audio_features.dtype != torch.float32:
            logging.info(f"Converting audio features from {audio_features.dtype} to float32")
            audio_features = audio_features.to(torch.float32)
                
        if video_features.dtype != torch.float32:
            logging.info(f"Converting video features from {video_features.dtype} to float32")
            video_features = video_features.to(torch.float32)
        
        # Check dimensions
        actual_audio_dim = audio_features.size(-1)
        actual_video_dim = video_features.size(-1)
        logging.info(f"Audio features dim: {actual_audio_dim}, expected: {self.audio_dim}")
        logging.info(f"Video features dim: {actual_video_dim}, expected: {self.video_dim}")
        
        # Recreate projections if dimensions don't match
        if actual_audio_dim != self.audio_dim:
            logging.warning(f"Audio feature dimension mismatch: {actual_audio_dim} vs expected {self.audio_dim}")
            logging.info(f"Recreating audio projection with correct dimensions: {actual_audio_dim} -> {self.output_dim}")
            self.audio_dim = actual_audio_dim
            self.audio_proj = nn.Linear(actual_audio_dim, self.output_dim).to(device=audio_features.device, dtype=torch.float32)
            
        if actual_video_dim != self.video_dim:
            logging.warning(f"Video feature dimension mismatch: {actual_video_dim} vs expected {self.video_dim}")
            logging.info(f"Recreating video projection with correct dimensions: {actual_video_dim} -> {self.output_dim}")
            self.video_dim = actual_video_dim
            self.video_proj = nn.Linear(actual_video_dim, self.output_dim).to(device=video_features.device, dtype=torch.float32)
            
        # Recreate fusion projection if needed
        concat_dim = actual_audio_dim + actual_video_dim
        if concat_dim != (self.audio_dim + self.video_dim):
            logging.warning(f"Concat dimension mismatch: {concat_dim} vs expected {self.audio_dim + self.video_dim}")
            logging.info(f"Recreating fusion projection with correct dimensions: {concat_dim} -> {self.output_dim}")
            self.fusion_proj = nn.Sequential(
                nn.Linear(concat_dim, self.output_dim),
                nn.ReLU()
            ).to(device=audio_features.device, dtype=torch.float32)
        
        # Ensure both features have compatible sequence dimensions
        batch_size = min(audio_features.size(0), video_features.size(0))
        seq_len = min(audio_features.size(1), video_features.size(1))
        
        # Truncate if necessary
        audio_features = audio_features[:batch_size, :seq_len]
        video_features = video_features[:batch_size, :seq_len]
        
        try:
            # Concatenate along the feature dimension
            concat_features = torch.cat([audio_features, video_features], dim=-1)
            
            # Apply projection
            logging.info(f"Applying fusion projection to tensor of shape {concat_features.shape}")
            
            # Ensure the linear projection has same dtype as input (float32)
            if self.fusion_proj[0].weight.dtype != torch.float32:
                logging.info(f"Converting fusion_proj weight from {self.fusion_proj[0].weight.dtype} to float32")
                self.fusion_proj = self.fusion_proj.to(torch.float32)
                
            fused_features = self.fusion_proj(concat_features)
            return fused_features
            
        except RuntimeError as e:
            logging.error(f"Error in fusion: {e}")
            logging.error(traceback.format_exc())
            
            # Fall back to audio only as a last resort
            logging.warning("Fusion failed, falling back to audio-only projection")
            
            # Ensure audio projection has same dtype as input
            if self.audio_proj.weight.dtype != torch.float32:
                logging.info(f"Converting audio_proj weight from {self.audio_proj.weight.dtype} to float32")
                self.audio_proj = self.audio_proj.to(torch.float32)
                
            return self.audio_proj(audio_features)


class SimpleFusion(nn.Module):
    """Simple fusion module that concatenates audio and video features and projects to output dimension"""
    
    def __init__(self, audio_dim, video_dim, output_dim, use_fp16=False):
        """
        Initialize the fusion module
        
        Args:
            audio_dim: Dimension of audio features
            video_dim: Dimension of video features
            output_dim: Dimension of output features
            use_fp16: Whether to use fp16 for better efficiency
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.output_dim = output_dim
        self.combined_dim = audio_dim + video_dim
        self.use_fp16 = use_fp16
        
        # Create projection layer
        dtype = torch.float16 if use_fp16 else torch.float32
        self.projection = nn.Linear(self.combined_dim, output_dim).to(dtype=dtype)
        
        logging.info(f"Initialized SimpleFusion: audio_dim={audio_dim}, video_dim={video_dim}, output_dim={output_dim}, use_fp16={use_fp16}")
    
    def forward(self, audio_features, video_features):
        """
        Fuse audio and video features
        
        Args:
            audio_features: Audio features [batch_size, seq_len, audio_dim]
            video_features: Video features [batch_size, seq_len, video_dim]
            
        Returns:
            Fused features [batch_size, seq_len, output_dim]
        """
        # Check if both inputs are provided
        if audio_features is None and video_features is None:
            raise ValueError("Both audio and video features are None")
        
        # Handle single modality case
        if audio_features is None:
            logging.info("Audio features are None, using video features only")
            return self.project_features(video_features)
        
        if video_features is None:
            logging.info("Video features are None, using audio features only")
            return self.project_features(audio_features)
        
        # Get batch size and sequence length
        batch_size = min(audio_features.size(0), video_features.size(0))
        seq_len = min(audio_features.size(1), video_features.size(1))
        
        # Truncate to match dimensions
        audio_features = audio_features[:batch_size, :seq_len]
        video_features = video_features[:batch_size, :seq_len]
        
        # Convert to fp16 if needed
        if self.use_fp16:
            if audio_features.dtype != torch.float16:
                audio_features = audio_features.to(torch.float16)
            if video_features.dtype != torch.float16:
                video_features = video_features.to(torch.float16)
        
        # Get actual dimensions
        actual_audio_dim = audio_features.size(-1)
        actual_video_dim = video_features.size(-1)
        
        # Check if dimensions match expected dimensions
        if actual_audio_dim != self.audio_dim or actual_video_dim != self.video_dim:
            logging.warning(f"Feature dimension mismatch: got audio={actual_audio_dim}, expected={self.audio_dim}, "
                           f"got video={actual_video_dim}, expected={self.video_dim}")
            
            # Update dimensions
            self.audio_dim = actual_audio_dim
            self.video_dim = actual_video_dim
            self.combined_dim = actual_audio_dim + actual_video_dim
            
            # Recreate projection if needed
            logging.info(f"Recreating projection layer: {self.combined_dim} -> {self.output_dim}")
            dtype = torch.float16 if self.use_fp16 else torch.float32
            self.projection = nn.Linear(self.combined_dim, self.output_dim).to(
                device=audio_features.device, dtype=dtype
            )
        
        # Concatenate along feature dimension
        try:
            combined = torch.cat([audio_features, video_features], dim=-1)
            # Log for debugging
            logging.info(f"Combined features: shape={combined.shape}, dtype={combined.dtype}")
            
            # Project to output dimension
            return self.project_features(combined)
        except RuntimeError as e:
            logging.error(f"Error in fusion: {e}")
            logging.error(traceback.format_exc())
            # Fall back to audio only
            logging.warning("Fusion failed, falling back to audio features only")
            return self.project_features(audio_features)
    
    def project_features(self, features):
        """Helper to project features to output dimension"""
        try:
            # Ensure data types match
            if self.use_fp16 and features.dtype != torch.float16:
                features = features.to(torch.float16)
            elif not self.use_fp16 and features.dtype != torch.float32:
                features = features.to(torch.float32)
                
            # Ensure projection is in the right data type
            if self.use_fp16 and self.projection.weight.dtype != torch.float16:
                self.projection = self.projection.to(torch.float16)
            elif not self.use_fp16 and self.projection.weight.dtype != torch.float32:
                self.projection = self.projection.to(torch.float32)
            
            # Get actual dimension
            actual_dim = features.size(-1)
            if actual_dim != self.combined_dim and actual_dim != self.audio_dim and actual_dim != self.video_dim:
                logging.warning(f"Feature dimension mismatch in projection: got {actual_dim}, expected {self.combined_dim}")
                
                # Create new projection with correct dimensions
                logging.info(f"Recreating projection layer: {actual_dim} -> {self.output_dim}")
                dtype = torch.float16 if self.use_fp16 else torch.float32
                self.projection = nn.Linear(actual_dim, self.output_dim).to(
                    device=features.device, dtype=dtype
                )
                self.combined_dim = actual_dim
                
            # Apply projection
            return self.projection(features)
        except RuntimeError as e:
            logging.error(f"Error in projection: {e}")
            logging.error(traceback.format_exc())
            
            # Get actual dimension and recreate projection if needed
            actual_dim = features.size(-1)
            if actual_dim != self.combined_dim:
                logging.warning(f"Feature dimension mismatch: got {actual_dim}, expected {self.combined_dim}")
                logging.info(f"Recreating projection layer: {actual_dim} -> {self.output_dim}")
                
                # Create new projection with correct dimensions
                dtype = torch.float16 if self.use_fp16 else torch.float32
                self.projection = nn.Linear(actual_dim, self.output_dim).to(
                    device=features.device, dtype=dtype
                )
                
                # Try again
                return self.projection(features)
            
            # If that's not the issue, re-raise
            raise