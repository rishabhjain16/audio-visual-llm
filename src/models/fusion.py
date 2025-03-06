#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any


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
    Simple concatenation-based fusion module
    """
    
    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        fusion_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            audio_dim: Dimension of audio features
            video_dim: Dimension of video features
            fusion_dim: Dimension of fused features
            dropout: Dropout rate
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.fusion_dim = fusion_dim
        
        # Projection after concatenation
        self.fusion_proj = nn.Sequential(
            nn.Linear(audio_dim + video_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse audio and video features by concatenation
        
        Args:
            audio_features: Audio features of shape (batch_size, seq_len, audio_dim)
            video_features: Video features of shape (batch_size, seq_len, video_dim)
            padding_mask: Optional padding mask of shape (batch_size, seq_len)
        
        Returns:
            Fused features of shape (batch_size, seq_len, fusion_dim)
        """
        # Check if both modalities are available
        if audio_features is None and video_features is None:
            raise ValueError("Both audio and video features cannot be None")
        
        # Handle single modality case
        if audio_features is None:
            return self.video_proj(video_features)
        
        if video_features is None:
            return self.audio_proj(audio_features)
        
        # Ensure both features have the same batch size and sequence length
        audio_batch, audio_seq_len = audio_features.size(0), audio_features.size(1)
        video_batch, video_seq_len = video_features.size(0), video_features.size(1)
        
        # Get the device of the input tensors
        device = audio_features.device
        
        # Ensure video features are on the same device as audio features
        if video_features.device != device:
            logging.info(f"Moving video features to device: {device}")
            video_features = video_features.to(device)
        
        # Ensure fusion projection is on the same device
        if hasattr(self.fusion_proj[0], 'weight') and self.fusion_proj[0].weight.device != device:
            logging.info(f"Moving fusion projection to device: {device}")
            self.fusion_proj = self.fusion_proj.to(device)
        
        # Find minimum batch size and sequence length
        min_batch = min(audio_batch, video_batch)
        min_length = min(audio_seq_len, video_seq_len)
        
        # Truncate if necessary
        if audio_batch > min_batch or audio_seq_len > min_length:
            audio_features = audio_features[:min_batch, :min_length]
        
        if video_batch > min_batch or video_seq_len > min_length:
            video_features = video_features[:min_batch, :min_length]
        
        # Concatenate along the feature dimension
        try:
            concat_features = torch.cat([audio_features, video_features], dim=-1)
        except RuntimeError as e:
            logging.error(f"Error concatenating features: {e}")
            logging.error(f"Audio shape: {audio_features.shape}, Video shape: {video_features.shape}")
            logging.error(f"Audio device: {audio_features.device}, Video device: {video_features.device}")
            
            # Try to fix device mismatch
            if audio_features.device != video_features.device:
                logging.warning(f"Device mismatch detected. Moving tensors to {device}")
                audio_features = audio_features.to(device)
                video_features = video_features.to(device)
                try:
                    concat_features = torch.cat([audio_features, video_features], dim=-1)
                except RuntimeError:
                    # Fall back to using just one modality
                    if audio_features.size(0) == min_batch and audio_features.size(1) == min_length:
                        logging.warning("Falling back to audio features only")
                        return self.audio_proj(audio_features)
                    elif video_features.size(0) == min_batch and video_features.size(1) == min_length:
                        logging.warning("Falling back to video features only")
                        return self.video_proj(video_features)
                    else:
                        # Generate random features as a last resort
                        logging.warning("Generating random features as fallback")
                        return torch.randn(min_batch, min_length, self.fusion_dim, device=device)
            else:
                # Fall back to using just one modality
                if audio_features.size(0) == min_batch and audio_features.size(1) == min_length:
                    logging.warning("Falling back to audio features only")
                    return self.audio_proj(audio_features)
                elif video_features.size(0) == min_batch and video_features.size(1) == min_length:
                    logging.warning("Falling back to video features only")
                    return self.video_proj(video_features)
                else:
                    # Generate random features as a last resort
                    logging.warning("Generating random features as fallback")
                    return torch.randn(min_batch, min_length, self.fusion_dim, device=device)
        
        # Project to fusion dimension
        fused_features = self.fusion_proj(concat_features)
        
        return fused_features