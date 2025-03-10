import torch
import torch.nn as nn
import logging
import math

class BaseModalityConnector(nn.Module):
    """Base class for modality connectors"""
    
    def __init__(self, input_dim, output_dim, device="cuda", dtype=torch.float32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        
    def forward(self, x):
        # Ensure input has correct dtype
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return self._forward_impl(x)
    
    def _forward_impl(self, x):
        raise NotImplementedError("Subclasses must implement _forward_impl")

class SimpleModalityConnector(BaseModalityConnector):
    """Original simple linear projection"""

    def __init__(self, input_dim, output_dim, device="cuda", dtype=torch.float32, max_seq_len=None, **kwargs):
        super().__init__(input_dim, output_dim, device, dtype)
        
        logging.info(f"Creating SimpleModalityConnector: input_dim={input_dim}, output_dim={output_dim}, dtype={dtype}")
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Use Xavier uniform initialization for better gradient flow
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        # Move to specified device and dtype
        self.linear = self.linear.to(device=device, dtype=dtype)
        
        logging.info(f"Created SimpleModalityConnector with xavier_uniform initialization")

    def _forward_impl(self, x):
        return self.linear(x) 

class DeepModalityConnector(BaseModalityConnector):
    """Multi-layer connector with residual connections and layer norm"""

    def __init__(self, input_dim, output_dim, device="cuda", dtype=torch.float32, hidden_dim=None, num_layers=2):
        super().__init__(input_dim, output_dim, device, dtype)
        
        hidden_dim = hidden_dim or max(input_dim, output_dim)
        
        logging.info(f"Creating DeepModalityConnector: input_dim={input_dim}, output_dim={output_dim}, "
                    f"hidden_dim={hidden_dim}, num_layers={num_layers}, dtype={dtype}")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_act = nn.GELU()
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers - 2):
            self.hidden_layers.append(nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ]))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Move to specified device and dtype
        self.to(device=device, dtype=dtype)
        
        logging.info(f"Created DeepModalityConnector with {num_layers} layers")

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use Xavier uniform initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _forward_impl(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_act(x)
        
        # Hidden layers with residual connections
        for linear, norm, act in self.hidden_layers:
            residual = x
            x = linear(x)
            x = norm(x)
            x = act(x)
            x = x + residual
        
        # Output projection
        x = self.output_proj(x)
        x = self.output_norm(x)
        
        return x

class ConvModalityConnector(BaseModalityConnector):
    """Connector that uses 1D convolutions to better capture sequence information"""

    def __init__(self, input_dim, output_dim, device="cuda", dtype=torch.float32, kernel_size=3):
        super().__init__(input_dim, output_dim, device, dtype)
        
        logging.info(f"Creating ConvModalityConnector: input_dim={input_dim}, output_dim={output_dim}, "
                    f"kernel_size={kernel_size}, dtype={dtype}")
        
        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        
        # Reshape projection: [B, S, C] -> [B, C, S] for conv1d
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(8, output_dim),  # More stable than LayerNorm for conv
            nn.GELU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(8, output_dim)
        )
        
        # Final projection layer
        self.final_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Move to specified device and dtype
        self.to(device=device, dtype=dtype)
        
        logging.info(f"Created ConvModalityConnector with kernel_size={kernel_size}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Use Xavier uniform for convolutional layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Use Xavier uniform for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def _forward_impl(self, x):
        # x shape: [B, S, C]
        # Transpose for conv1d: [B, C, S]
        x_conv = x.transpose(1, 2)
        
        # Apply conv layers
        x_conv = self.conv_layers(x_conv)
        
        # Transpose back: [B, S, C]
        x = x_conv.transpose(1, 2)
        
        # Final projection
        x = self.final_proj(x)
        x = self.norm(x)
        
        return x

class AttentionModalityConnector(BaseModalityConnector):
    """Connector that uses self-attention to better handle long sequences"""

    def __init__(self, input_dim, output_dim, device="cuda", dtype=torch.float32, heads=8):
        super().__init__(input_dim, output_dim, device, dtype)
        
        logging.info(f"Creating AttentionModalityConnector: input_dim={input_dim}, output_dim={output_dim}, "
                    f"heads={heads}, dtype={dtype}")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.norm1 = nn.LayerNorm(output_dim)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(output_dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim)
        )
        self.norm3 = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Move to specified device and dtype
        self.to(device=device, dtype=dtype)
        
        logging.info(f"Created AttentionModalityConnector with {heads} attention heads")

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use Xavier uniform initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _forward_impl(self, x):
        # Input projection
        residual = x
        x = self.input_proj(x)
        x = self.norm1(x)
        
        # Self-attention with residual
        residual = x
        x, _ = self.attention(x, x, x)
        x = x + residual
        x = self.norm2(x)
        
        # Feed-forward with residual
        residual = x
        x = self.ff(x)
        x = x + residual
        x = self.norm3(x)
        
        return x
        
class AdaptiveModalityConnector(BaseModalityConnector):
    """Connector that adapts to sequence length using pooling"""

    def __init__(self, input_dim, output_dim, device="cuda", dtype=torch.float32, max_seq_len=1536):
        super().__init__(input_dim, output_dim, device, dtype)
        
        logging.info(f"Creating AdaptiveModalityConnector: input_dim={input_dim}, output_dim={output_dim}, "
                    f"max_seq_len={max_seq_len}, dtype={dtype}")
        
        self.max_seq_len = max_seq_len
        mid_dim = (input_dim + output_dim) // 2
        
        # Embedding improvement
        self.input_proj = nn.Linear(input_dim, mid_dim)
        self.norm1 = nn.LayerNorm(mid_dim)
        self.act = nn.GELU()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(mid_dim, max_len=max_seq_len)
        
        # Sequence length adaptive layer
        self.adaptive_pool = AdaptiveSequencePooling(mid_dim)
        
        # Output projection
        self.output_proj = nn.Linear(mid_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Move to specified device and dtype
        self.to(device=device, dtype=dtype)
        
        logging.info(f"Created AdaptiveModalityConnector with max_seq_len={max_seq_len}")

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use Xavier uniform initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _forward_impl(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.norm1(x)
        x = self.act(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Adaptive pooling based on sequence length
        x = self.adaptive_pool(x)
        
        # Output projection
        x = self.output_proj(x)
        x = self.norm2(x)
        
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
        
class AdaptiveSequencePooling(nn.Module):
    """Adaptive pooling module for handling varying sequence lengths"""
    
    def __init__(self, dim):
        super().__init__()
        
        # Long sequences: use strided convolution for downsampling
        self.long_adapter = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        )
        
        # Attention for context mixing
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(dim)
        
        # Initialize weights with Xavier uniform
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Apply different strategies based on sequence length
        if seq_len > 512:
            # For long sequences, apply convolutional downsampling
            # Transpose for conv1d: [B, C, S]
            x_conv = x.transpose(1, 2)
            x_conv = self.long_adapter(x_conv)
            # Transpose back: [B, S, C]
            x = x_conv.transpose(1, 2)
            
        # Apply attention for all sequences
        residual = x
        x, _ = self.attn(x, x, x)
        x = x + residual
        x = self.norm(x)
        
        return x

# Factory function to create the appropriate connector
def create_modality_connector(connector_type, input_dim, output_dim, **kwargs):
    """Create a modality connector of the specified type"""
    
    connector_map = {
        "simple": SimpleModalityConnector,
        "deep": DeepModalityConnector,
        "conv": ConvModalityConnector,
        "attention": AttentionModalityConnector,
        "adaptive": AdaptiveModalityConnector
    }
    
    if connector_type not in connector_map:
        logging.warning(f"Unknown connector type '{connector_type}', using 'deep' instead")
        connector_type = "deep"
    
    connector_class = connector_map[connector_type]
    return connector_class(input_dim, output_dim, **kwargs)

# For backward compatibility
ModalityConnector = SimpleModalityConnector 