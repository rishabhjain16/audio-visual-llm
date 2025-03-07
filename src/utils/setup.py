#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import logging
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from typing import Optional, Union


def setup_logging(log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Setup loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Set root logger to WARNING to reduce noise
    
    # Set up project logger
    project_logger = logging.getLogger('src')
    project_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce logging for specific verbose modules
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('peft').setLevel(logging.WARNING)
    
    # Log configuration
    project_logger.info(f"Logging initialized with level={level}")
    if log_file is not None:
        project_logger.info(f"Logging to {log_file}")


def setup_seed(seed: int = 42):
    """
    Setup random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic for reproducibility
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    logging.info(f"Random seed set to {seed}")


def setup_environment():
    """
    Setup environment variables and PyTorch configuration
    """
    # Set environment variables for better performance
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Disable grad for faster evaluation if not training
    torch.set_grad_enabled(False)
    
    # Enable tensor cores for better performance
    torch.set_float32_matmul_precision("high")
    
    # Log PyTorch and CUDA information
    logging.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"CUDA devices available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
        
        # Log available memory for main GPU
        if torch.cuda.device_count() > 0:
            main_gpu = torch.cuda.current_device()
            free_memory, total_memory = torch.cuda.mem_get_info(main_gpu)
            free_memory_gb = free_memory / (1024 ** 3)
            total_memory_gb = total_memory / (1024 ** 3)
            logging.info(f"GPU {main_gpu} memory: {free_memory_gb:.2f} GB free / {total_memory_gb:.2f} GB total")
    else:
        logging.warning("CUDA is not available. Running on CPU only.")


def setup_amp(fp16: bool = True):
    """
    Setup automatic mixed precision
    
    Args:
        fp16: Whether to use mixed precision training
    """
    if fp16:
        logging.info("Using mixed precision (FP16)")
    else:
        logging.info("Using full precision (FP32)")
    
    return fp16


def optimize_memory():
    """
    Apply memory optimization techniques
    """
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Disable gradient computation for inference
    torch.set_grad_enabled(False)
    
    # Use memory efficient attention if available
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        # This flag enables memory-efficient attention in PyTorch 2.0+
        os.environ["PYTORCH_ENABLE_MEM_EFFICIENT_SDPA"] = "1"
    
    logging.info("Applied memory optimizations") 