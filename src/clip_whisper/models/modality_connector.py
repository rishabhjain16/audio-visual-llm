import torch
import torch.nn as nn
import logging

class ModalityConnector(nn.Module):
    """Linear projection for modality encoding to LLM dimension"""

    def __init__(self, input_dim, output_dim, device="cuda", dtype=torch.float32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        
        logging.info(f"Creating ModalityConnector: input_dim={input_dim}, output_dim={output_dim}, dtype={dtype}")
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Use standard PyTorch initialization 
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.bias)
        
        # Move to specified device and dtype
        self.linear = self.linear.to(device=device, dtype=dtype)
        
        logging.info(f"Created ModalityConnector with standard initialization")

    def forward(self, x):
        # Ensure input has correct dtype
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return self.linear(x) 