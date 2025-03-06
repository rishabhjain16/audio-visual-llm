#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import logging
import torch
import traceback

from .avsr_llm import AVSRLLM

logger = logging.getLogger(__name__)

def create_model_from_config(config):
    """
    Create a model from configuration
    
    Args:
        config: Configuration object or dictionary
        
    Returns:
        model: Initialized model
    """
    try:
        logger.info("Creating model from configuration")
        
        # Get device
        device = getattr(config, 'device', "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create AVSRLLM model
        model = AVSRLLM(
            config=config,
            device=device
        )
        
        logger.info(f"Created AVSRLLM model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        logger.error(traceback.format_exc())
        raise 