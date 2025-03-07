import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class VisualFrontend(nn.Module):
    """Visual frontend for lip feature extraction"""
    def __init__(self, in_channels=3, out_channels=512):
        super().__init__()
        
        self.frontend = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(256, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
    
    def forward(self, x):
        """Forward pass for 3D video input
        
        Args:
            x: Input tensor of shape (batch_size, channels, time, height, width)
            
        Returns:
            Visual features of shape (batch_size, time, features)
        """
        # (batch, channels, time, height, width) -> (batch, features, time, h, w)
        x = self.frontend(x)
        
        # Global average pooling over spatial dimensions (h, w)
        x = torch.mean(x, dim=(-1, -2))
        
        # (batch, features, time) -> (batch, time, features)
        x = x.transpose(1, 2)
        
        return x

class LandmarkEncoder(nn.Module):
    """Encoder for landmark features"""
    def __init__(self, input_dim, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
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
        """Forward pass for landmark features
        
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
        
        # Create transformer mask (for padding)
        if mask is not None:
            # Transformer expects mask where 0 is value to attend to, 1 is value to ignore
            transformer_mask = mask
        else:
            transformer_mask = None
        
        # Apply transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        """Add positional encoding to input
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class VisualEncoder(nn.Module):
    """Full visual encoder for AVSR"""
    def __init__(self, 
                 input_type="video",  # "video" or "landmarks"
                 landmark_dim=60,     # dimension of landmark features
                 hidden_dim=512,      # encoder hidden dimension
                 num_layers=6,        # number of transformer layers
                 dropout=0.1):
        super().__init__()
        
        self.input_type = input_type
        
        if input_type == "video":
            # For raw video input
            self.frontend = VisualFrontend(in_channels=3, out_channels=hidden_dim)
        else:
            # For pre-extracted landmark features
            self.landmark_encoder = LandmarkEncoder(
                input_dim=landmark_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
    
    def forward(self, x, mask=None):
        """Forward pass
        
        Args:
            x: Input tensor - either video (B,C,T,H,W) or landmarks (B,T,D)
            mask: Padding mask for transformer
            
        Returns:
            Encoded visual features (B,T,hidden_dim)
        """
        if self.input_type == "video":
            # Process video input
            features = self.frontend(x)
        else:
            # Process landmark input
            features = self.landmark_encoder(x, mask)
        
        return features