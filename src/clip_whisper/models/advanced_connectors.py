import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

class CrossModalConnector(nn.Module):
    """
    Cross-Modal Attention Connector that enables audio and video modalities to attend to each other.
    Inspired by cross-attention mechanisms in papers like:
    - "Audio-Visual Speech Recognition with a Hybrid CTC/Attention Architecture" (Afouras et al.)
    - "Multimodal Transformers for Speech Recognition" (Xu et al.)
    """
    
    def __init__(
        self, 
        audio_dim, 
        video_dim, 
        output_dim, 
        device="cuda", 
        dtype=torch.float32,
        num_heads=8,
        dropout=0.1,
        num_layers=2
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.num_heads = num_heads
        
        # Common dimensions for cross-attention
        self.common_dim = max(256, output_dim // 2)
        
        # Initial projections to common dimension
        self.audio_proj = nn.Linear(audio_dim, self.common_dim)
        self.video_proj = nn.Linear(video_dim, self.common_dim)
        
        # Normalization layers
        self.audio_norm = nn.LayerNorm(self.common_dim)
        self.video_norm = nn.LayerNorm(self.common_dim)
        
        # Cross-attention layers
        self.cross_layers = nn.ModuleList([
            CrossModalLayer(
                dim=self.common_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final projection to output dimension
        self.output_proj = nn.Linear(self.common_dim * 2, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device and dtype
        self.to(device=device, dtype=dtype)
        
        logging.info(f"Created CrossModalConnector: audio_dim={audio_dim}, video_dim={video_dim}, "
                    f"output_dim={output_dim}, num_heads={num_heads}, num_layers={num_layers}")
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, audio_features, video_features, audio_mask=None, video_mask=None):
        """
        Forward pass of the cross-modal connector
        
        Args:
            audio_features: Tensor of shape [batch_size, audio_seq_len, audio_dim]
            video_features: Tensor of shape [batch_size, video_seq_len, video_dim]
            audio_mask: Optional mask for audio sequence (1=valid, 0=padding)
            video_mask: Optional mask for video sequence (1=valid, 0=padding)
            
        Returns:
            Tensor of shape [batch_size, max_seq_len, output_dim]
        """
        batch_size = audio_features.shape[0]
        
        # Ensure inputs have correct dtype
        if audio_features.dtype != self.dtype:
            audio_features = audio_features.to(self.dtype)
        if video_features.dtype != self.dtype:
            video_features = video_features.to(self.dtype)
        
        # Project to common dimension
        audio_proj = self.audio_norm(self.audio_proj(audio_features))
        video_proj = self.video_norm(self.video_proj(video_features))
        
        # Process through cross-modal layers
        for layer in self.cross_layers:
            audio_proj, video_proj = layer(
                audio_proj, video_proj, audio_mask, video_mask
            )
        
        # Use the longest sequence length
        max_seq_len = max(audio_proj.size(1), video_proj.size(1))
        
        # Pad sequences to match the longest one
        if audio_proj.size(1) < max_seq_len:
            audio_proj = F.pad(
                audio_proj, 
                (0, 0, 0, max_seq_len - audio_proj.size(1)),
                "constant", 0
            )
        if video_proj.size(1) < max_seq_len:
            video_proj = F.pad(
                video_proj, 
                (0, 0, 0, max_seq_len - video_proj.size(1)),
                "constant", 0
            )
        
        # Concatenate features and project to output dimension
        fused_features = torch.cat([audio_proj, video_proj], dim=-1)
        output = self.output_norm(self.output_proj(fused_features))
        
        return output


class CrossModalLayer(nn.Module):
    """A cross-modal attention layer that enables two modalities to attend to each other"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Audio attends to video
        self.audio_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.audio_norm1 = nn.LayerNorm(dim)
        self.audio_norm2 = nn.LayerNorm(dim)
        self.audio_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Video attends to audio
        self.video_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.video_norm1 = nn.LayerNorm(dim)
        self.video_norm2 = nn.LayerNorm(dim)
        self.video_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, audio, video, audio_mask=None, video_mask=None):
        # Convert masks to attention format if provided
        audio_attn_mask = None
        video_attn_mask = None
        
        if audio_mask is not None and video_mask is not None:
            # Create attention masks (additive masks for attention mechanism)
            audio_attn_mask = ~(video_mask.bool()).to(audio.device)
            video_attn_mask = ~(audio_mask.bool()).to(video.device)
        
        # Audio attends to video (cross-attention)
        audio_residual = audio
        audio_out, _ = self.audio_attn(
            query=audio,
            key=video,
            value=video,
            key_padding_mask=audio_attn_mask
        )
        audio = self.audio_norm1(audio_out + audio_residual)
        
        # Audio MLP with residual
        audio_residual = audio
        audio = self.audio_mlp(audio)
        audio = self.audio_norm2(audio + audio_residual)
        
        # Video attends to audio (cross-attention)
        video_residual = video
        video_out, _ = self.video_attn(
            query=video,
            key=audio,
            value=audio,
            key_padding_mask=video_attn_mask
        )
        video = self.video_norm1(video_out + video_residual)
        
        # Video MLP with residual
        video_residual = video
        video = self.video_mlp(video)
        video = self.video_norm2(video + video_residual)
        
        return audio, video


class QformerConnector(nn.Module):
    """
    Qformer-style connector for audio-visual data, inspired by:
    - "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (Li et al.)
    - "Flamingo: a Visual Language Model for Few-Shot Learning" (Alayrac et al.)
    
    Uses a set of learnable query vectors to extract information from modality-specific features
    through cross-attention.
    """
    
    def __init__(
        self, 
        audio_dim, 
        video_dim, 
        output_dim, 
        device="cuda", 
        dtype=torch.float32,
        num_queries=32,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.num_queries = num_queries
        
        # Common dimensions for processing
        self.common_dim = max(256, output_dim // 2)
        
        # Learnable query vectors for extraction
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, self.common_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Initial projections to common dimension
        self.audio_proj = nn.Linear(audio_dim, self.common_dim)
        self.video_proj = nn.Linear(video_dim, self.common_dim)
        
        # Normalization layers
        self.audio_norm = nn.LayerNorm(self.common_dim)
        self.video_norm = nn.LayerNorm(self.common_dim)
        
        # QFormer blocks to process queries
        self.query_encoder = nn.ModuleList([
            QformerBlock(
                dim=self.common_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final projection to output dimension
        self.output_proj = nn.Linear(self.common_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device and dtype
        self.to(device=device, dtype=dtype)
        
        logging.info(f"Created QformerConnector: audio_dim={audio_dim}, video_dim={video_dim}, "
                    f"output_dim={output_dim}, num_queries={num_queries}, num_layers={num_layers}")
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, audio_features, video_features, audio_mask=None, video_mask=None):
        """
        Forward pass of the Qformer connector
        
        Args:
            audio_features: Tensor of shape [batch_size, audio_seq_len, audio_dim]
            video_features: Tensor of shape [batch_size, video_seq_len, video_dim]
            audio_mask: Optional mask for audio sequence (1=valid, 0=padding)
            video_mask: Optional mask for video sequence (1=valid, 0=padding)
            
        Returns:
            Tensor of shape [batch_size, num_queries, output_dim]
        """
        batch_size = audio_features.shape[0]
        
        # Ensure inputs have correct dtype
        if audio_features.dtype != self.dtype:
            audio_features = audio_features.to(self.dtype)
        if video_features.dtype != self.dtype:
            video_features = video_features.to(self.dtype)
        
        # Project features to common dimension
        audio_proj = self.audio_norm(self.audio_proj(audio_features))
        video_proj = self.video_norm(self.video_proj(video_features))
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Process query tokens through Qformer blocks
        for layer in self.query_encoder:
            query_tokens = layer(
                query_tokens, 
                audio_proj, 
                video_proj,
                audio_mask, 
                video_mask
            )
        
        # Project to output dimension
        output = self.output_norm(self.output_proj(query_tokens))
        
        return output


class QformerBlock(nn.Module):
    """
    A Qformer block that processes query tokens through self-attention and 
    cross-attention with audio and video features.
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Self-attention for query tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        
        # Cross-attention for attending to audio
        self.audio_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        
        # Cross-attention for attending to video
        self.video_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm3 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(dim)
    
    def forward(self, query, audio, video, audio_mask=None, video_mask=None):
        # Self-attention
        residual = query
        query_out, _ = self.self_attn(query, query, query)
        query = self.norm1(query_out + residual)
        
        # Cross-attention with audio
        residual = query
        query_out, _ = self.audio_attn(
            query=query,
            key=audio,
            value=audio,
            key_padding_mask=None if audio_mask is None else ~audio_mask
        )
        query = self.norm2(query_out + residual)
        
        # Cross-attention with video
        residual = query
        query_out, _ = self.video_attn(
            query=query,
            key=video,
            value=video,
            key_padding_mask=None if video_mask is None else ~video_mask
        )
        query = self.norm3(query_out + residual)
        
        # Feed-forward network
        residual = query
        query = self.mlp(query)
        query = self.norm4(query + residual)
        
        return query


class MultimodalPerceiverConnector(nn.Module):
    """
    Perceiver-IO based connector for audio-visual data, inspired by:
    - "Perceiver IO: A General Architecture for Structured Inputs & Outputs" (Jaegle et al.)
    
    Processes long audio-visual sequences through a small set of latent vectors, making it
    particularly efficient for long sequences while preserving cross-modal relationships.
    """
    
    def __init__(
        self, 
        audio_dim, 
        video_dim, 
        output_dim, 
        device="cuda", 
        dtype=torch.float32,
        num_latents=64,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.num_latents = num_latents
        
        # Common dimensions for processing
        self.common_dim = max(256, output_dim // 2)
        
        # Learnable latent vectors
        self.latent_vectors = nn.Parameter(torch.zeros(1, num_latents, self.common_dim))
        nn.init.trunc_normal_(self.latent_vectors, std=0.02)
        
        # Initial projections to common dimension
        self.audio_proj = nn.Linear(audio_dim, self.common_dim)
        self.video_proj = nn.Linear(video_dim, self.common_dim)
        
        # Position embeddings for audio and video
        self.audio_pos_embed = PositionalEncoding(self.common_dim, max_len=2000)
        self.video_pos_embed = PositionalEncoding(self.common_dim, max_len=2000)
        
        # Cross-attention layers for latents attending to inputs
        self.input_processors = nn.ModuleList([
            PerceiverAttentionBlock(
                dim=self.common_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Self-attention layers for processing latents
        self.latent_processors = nn.ModuleList([
            SelfAttentionBlock(
                dim=self.common_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final projection to output dimension
        self.output_proj = nn.Linear(self.common_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device and dtype
        self.to(device=device, dtype=dtype)
        
        logging.info(f"Created MultimodalPerceiverConnector: audio_dim={audio_dim}, video_dim={video_dim}, "
                    f"output_dim={output_dim}, num_latents={num_latents}, num_layers={num_layers}")
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, audio_features, video_features, audio_mask=None, video_mask=None):
        """
        Forward pass of the Perceiver connector
        
        Args:
            audio_features: Tensor of shape [batch_size, audio_seq_len, audio_dim]
            video_features: Tensor of shape [batch_size, video_seq_len, video_dim]
            audio_mask: Optional mask for audio sequence (1=valid, 0=padding)
            video_mask: Optional mask for video sequence (1=valid, 0=padding)
            
        Returns:
            Tensor of shape [batch_size, output_seq_len, output_dim]
        """
        batch_size = audio_features.shape[0]
        
        # Ensure inputs have correct dtype
        if audio_features.dtype != self.dtype:
            audio_features = audio_features.to(self.dtype)
        if video_features.dtype != self.dtype:
            video_features = video_features.to(self.dtype)
        
        # Project features to common dimension and add positional embeddings
        audio_proj = self.audio_pos_embed(self.audio_proj(audio_features))
        video_proj = self.video_pos_embed(self.video_proj(video_features))
        
        # Combine audio and video features
        combined_features = torch.cat([audio_proj, video_proj], dim=1)
        combined_mask = None
        if audio_mask is not None and video_mask is not None:
            combined_mask = torch.cat([audio_mask, video_mask], dim=1)
        
        # Expand latent vectors to batch size
        latents = self.latent_vectors.expand(batch_size, -1, -1)
        
        # Process through perceiver blocks
        for i in range(len(self.input_processors)):
            # Cross-attention: latents attend to inputs
            latents = self.input_processors[i](
                latents, combined_features, attention_mask=combined_mask
            )
            
            # Self-attention: latents attend to latents
            latents = self.latent_processors[i](latents)
        
        # Project to output dimension
        output = self.output_norm(self.output_proj(latents))
        
        return output


class PerceiverAttentionBlock(nn.Module):
    """Cross-attention block for latents attending to inputs"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, latents, inputs, attention_mask=None):
        # Cross-attention
        residual = latents
        latents_out, _ = self.attn(
            query=latents,
            key=inputs,
            value=inputs,
            key_padding_mask=None if attention_mask is None else ~attention_mask
        )
        latents = self.norm1(latents_out + residual)
        
        # Feed-forward
        residual = latents
        latents = self.mlp(latents)
        latents = self.norm2(latents + residual)
        
        return latents


class SelfAttentionBlock(nn.Module):
    """Self-attention block for latents attending to themselves"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Self-attention
        residual = x
        x_out, _ = self.attn(x, x, x)
        x = self.norm1(x_out + residual)
        
        # Feed-forward
        residual = x
        x = self.mlp(x)
        x = self.norm2(x + residual)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence models"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0) 