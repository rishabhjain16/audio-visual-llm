import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class AudioEncoder(nn.Module):
    """Audio encoder for AVSR"""
    def __init__(self,
                 input_dim=80,       # mel spectrogram features
                 hidden_dim=512,     # encoder hidden dimension
                 num_layers=6,       # number of transformer layers
                 dropout=0.1):
        super().__init__()
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        from .visual_encoder import PositionalEncoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
    
    def forward(self, x, mask=None):
        """Forward pass for audio features
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Padding mask where True indicates padding (batch_size, seq_len)
            
        Returns:
            Encoded features of shape (batch_size, seq_len, hidden_dim)
        """
        # Embed features
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        return output

class AudioCNN(nn.Module):
    """CNN-based audio encoder for AVSR"""
    def __init__(self,
                 input_dim=80,       # mel spectrogram features
                 hidden_dim=512,      # encoder hidden dimension
                 num_layers=4,        # number of CNN layers
                 dropout=0.1):
        super().__init__()
        
        layers = []
        in_channels = 1  # Start with 1 channel (batch, 1, time, freq)
        out_channels = 32
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 2)) if i < num_layers - 1 else nn.Identity()
            ])
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)
        
        self.cnn = nn.Sequential(*layers)
        
        # Calculate the output size
        freq_dim = input_dim // (2 ** (num_layers - 1))
        
        # Linear projection to hidden_dim
        self.projection = nn.Linear(in_channels * freq_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """Forward pass for audio features
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Encoded features of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Reshape to (batch, channels, time, freq)
        x = x.unsqueeze(1)
        
        # Apply CNN
        x = self.cnn(x)
        
        # Reshape to (batch, time, channels * freq)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, -1)
        
        # Project to hidden_dim
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x