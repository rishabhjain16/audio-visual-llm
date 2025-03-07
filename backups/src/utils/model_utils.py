#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import shutil
from safetensors.torch import save_file, load_file


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
        strict: Whether to strictly enforce that the keys in state_dict match model
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint based on file extension
    if checkpoint_path.suffix == ".safetensors":
        # Load from safetensors format
        state_dict = load_file(checkpoint_path, device=device)
        metadata = {}  # Safetensors doesn't store metadata
    else:
        # Load from PyTorch format
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            metadata = {k: v for k, v in checkpoint.items() if k != "model"}
        else:
            state_dict = checkpoint
            metadata = {}
    
    # Handle prefix in state dict keys
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    # Load state dict
    if hasattr(model, "load_state_dict"):
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            logging.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {unexpected_keys}")
    
    logging.info(f"Checkpoint loaded successfully")
    
    return metadata


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metric: Optional[float] = None,
    output_path: Union[str, Path] = "checkpoints/model.pt",
    metadata: Optional[Dict[str, Any]] = None,
    use_safetensors: bool = False
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        epoch: Current epoch
        step: Current training step
        metric: Current evaluation metric
        output_path: Path to save checkpoint
        metadata: Additional metadata to save
        use_safetensors: Whether to use safetensors format
    """
    output_path = Path(output_path)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get model state dict
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    # Create checkpoint dictionary
    checkpoint = {
        "state_dict": state_dict
    }
    
    # Add optimizer state
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    
    # Add scheduler state
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    
    # Add training metadata
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if step is not None:
        checkpoint["step"] = step
    if metric is not None:
        checkpoint["metric"] = metric
    
    # Add additional metadata
    if metadata is not None:
        checkpoint.update(metadata)
    
    # Save checkpoint
    if use_safetensors:
        # Convert non-tensor values to metadata for safetensors
        tensors_dict = {}
        additional_metadata = {}
        
        for k, v in checkpoint.items():
            if isinstance(v, torch.Tensor):
                tensors_dict[k] = v
            elif k == "state_dict":
                tensors_dict.update(v)
            else:
                # Convert to JSON string for metadata
                additional_metadata[k] = json.dumps(v, default=str)
        
        # Save using safetensors
        save_file(tensors_dict, output_path, metadata=additional_metadata)
    else:
        # Save using PyTorch
        torch.save(checkpoint, output_path)
    
    logging.info(f"Checkpoint saved to {output_path}")


def load_huggingface_model(
    model_name_or_path: str,
    model_class: Any,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    fp16: bool = False,
    quantization: Optional[str] = None
) -> Any:
    """
    Load a Hugging Face model
    
    Args:
        model_name_or_path: Model name or path
        model_class: Model class to instantiate
        cache_dir: Cache directory for downloading models
        device: Device to load model to
        fp16: Whether to use half precision
        quantization: Quantization method (e.g., "int8", "int4")
        
    Returns:
        Loaded model
    """
    from transformers import AutoConfig
    from transformers.utils import cached_file
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info(f"Loading HuggingFace model: {model_name_or_path}")
    
    # Setup quantization if specified
    if quantization:
        try:
            from transformers import BitsAndBytesConfig
            
            if quantization == "int8":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
            elif quantization == "int4":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                logging.warning(f"Unknown quantization method: {quantization}, using default")
                quantization_config = None
                
        except ImportError:
            logging.warning("bitsandbytes not installed, quantization disabled")
            quantization_config = None
    else:
        quantization_config = None
    
    # Load model
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    
    if fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Instantiate model
    model = model_class.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if device != "auto" and not quantization:
        model = model.to(device)
    
    logging.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    return model


def freeze_model_layers(
    model: nn.Module,
    freeze_layer_names: Optional[List[str]] = None,
    unfreeze_layer_names: Optional[List[str]] = None,
    freeze_layer_types: Optional[List[type]] = None,
    unfreeze_layer_types: Optional[List[type]] = None,
) -> nn.Module:
    """
    Freeze or unfreeze model layers
    
    Args:
        model: Model to freeze/unfreeze
        freeze_layer_names: List of layer names to freeze
        unfreeze_layer_names: List of layer names to unfreeze
        freeze_layer_types: List of layer types to freeze
        unfreeze_layer_types: List of layer types to unfreeze
        
    Returns:
        Model with frozen/unfrozen layers
    """
    # Initialize counters
    frozen_params = 0
    total_params = 0
    
    # Default to freezing everything if nothing is specified
    if (freeze_layer_names is None and unfreeze_layer_names is None and 
        freeze_layer_types is None and unfreeze_layer_types is None):
        logging.warning("No freeze/unfreeze specifications provided, doing nothing")
        return model
    
    # Process all named parameters
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # Get parent module name
        parts = name.split('.')
        module_name = '.'.join(parts[:-1])
        
        # Find parent module
        module = model
        for part in module_name.split('.'):
            if part:
                module = getattr(module, part)
        
        should_freeze = False
        
        # Check layer names
        if freeze_layer_names is not None:
            for freeze_name in freeze_layer_names:
                if freeze_name in name:
                    should_freeze = True
                    break
        
        # Check layer types
        if freeze_layer_types is not None and not should_freeze:
            if any(isinstance(module, layer_type) for layer_type in freeze_layer_types):
                should_freeze = True
        
        # Check unfreeze names (overrides freeze)
        if unfreeze_layer_names is not None and should_freeze:
            for unfreeze_name in unfreeze_layer_names:
                if unfreeze_name in name:
                    should_freeze = False
                    break
        
        # Check unfreeze types (overrides freeze)
        if unfreeze_layer_types is not None and should_freeze:
            if any(isinstance(module, layer_type) for layer_type in unfreeze_layer_types):
                should_freeze = False
        
        # Apply freezing
        if should_freeze:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
    
    # Log summary
    if total_params > 0:
        frozen_percentage = (frozen_params / total_params) * 100
        logging.info(f"Frozen {frozen_params:,} parameters out of {total_params:,} ({frozen_percentage:.2f}%)")
    
    return model


def get_module_state_summary(model: nn.Module) -> str:
    """
    Get summary of model module states (frozen vs trainable)
    
    Args:
        model: Model to summarize
        
    Returns:
        String summary of model module states
    """
    module_states = {}
    
    # Process all named parameters
    for name, param in model.named_parameters():
        # Get parent module name
        parts = name.split('.')
        if len(parts) > 1:
            module_name = '.'.join(parts[:-1])
        else:
            module_name = "root"
        
        # Initialize module state if not exist
        if module_name not in module_states:
            module_states[module_name] = {
                "trainable_params": 0,
                "frozen_params": 0,
                "total_params": 0
            }
        
        # Update module state
        param_count = param.numel()
        module_states[module_name]["total_params"] += param_count
        
        if param.requires_grad:
            module_states[module_name]["trainable_params"] += param_count
        else:
            module_states[module_name]["frozen_params"] += param_count
    
    # Create summary string
    summary = []
    summary.append("Model module states:")
    
    # Calculate overall stats
    total_trainable = sum(state["trainable_params"] for state in module_states.values())
    total_frozen = sum(state["frozen_params"] for state in module_states.values())
    total_params = sum(state["total_params"] for state in module_states.values())
    
    # Add overall summary
    if total_params > 0:
        trainable_percentage = (total_trainable / total_params) * 100
        summary.append(f"  Overall: {total_trainable:,} trainable parameters, {total_frozen:,} frozen parameters ({trainable_percentage:.2f}% trainable)")
    
    # Add module-specific summaries
    for module_name, state in sorted(module_states.items()):
        if state["total_params"] > 0:
            trainable_percentage = (state["trainable_params"] / state["total_params"]) * 100
            summary.append(f"  {module_name}: {state['trainable_params']:,} trainable, {state['frozen_params']:,} frozen ({trainable_percentage:.2f}% trainable)")
    
    return "\n".join(summary) 