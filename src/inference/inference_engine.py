#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from src.models.avsr_llm import AVSRLLM
from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.audio_processor import AudioProcessor
from src.utils.config import AVSRConfig
from src.utils.setup import setup_logging


class AVSRInferenceEngine:
    """
    Inference engine for AVSR-LLM model
    """
    def __init__(
        self,
        config: AVSRConfig,
        checkpoint_path: str,
        device: str = "cuda",
        max_tokens: int = 1024,
    ):
        """
        Initialize the inference engine
        
        Args:
            config: Configuration object
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            max_tokens: Maximum number of tokens to generate
        """
        self.config = config
        self.device = device
        self.max_tokens = max_tokens
        
        logging.info(f"Initializing inference engine with device: {device}")
        
        # Initialize processors
        self.video_processor = VideoProcessor(config)
        self.audio_processor = AudioProcessor(config)
        
        # Load model
        self.load_model(checkpoint_path)
        
    def load_model(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        logging.info(f"Loading model from {checkpoint_path}")
        
        # Load model using the from_checkpoint class method
        self.model = AVSRLLM.from_checkpoint(checkpoint_path, device=self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        logging.info("Model loaded successfully")
    
    def preprocess_video(self, video_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess video for inference
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed video features
        """
        return self.video_processor.extract_features(video_path).to(self.device)
    
    def preprocess_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess audio for inference
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio features
        """
        return self.audio_processor.extract_features(audio_path).to(self.device)
    
    @torch.no_grad()
    def transcribe(
        self,
        video: Optional[Union[str, Path, torch.Tensor]] = None,
        audio: Optional[Union[str, Path, torch.Tensor]] = None,
        beam_size: int = 5,
        max_length: int = 100,
        min_length: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Transcribe audio/video
        
        Args:
            video: Video input (path or tensor)
            audio: Audio input (path or tensor)
            beam_size: Beam size for beam search decoding
            max_length: Maximum length of generated sequence
            min_length: Minimum length of generated sequence
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            length_penalty: Length penalty
            
        Returns:
            Dictionary containing transcription and metadata
        """
        if video is None and audio is None:
            raise ValueError("Either video or audio must be provided")
        
        # Preprocess inputs
        if video is not None:
            if isinstance(video, (str, Path)):
                video_features = self.preprocess_video(video)
            else:
                video_features = video.to(self.device)
        else:
            video_features = None
        
        if audio is not None:
            if isinstance(audio, (str, Path)):
                audio_features = self.preprocess_audio(audio)
            else:
                audio_features = audio.to(self.device)
        else:
            audio_features = None
        
        # Ensure features have batch dimension
        if video_features is not None and video_features.dim() == 3:
            video_features = video_features.unsqueeze(0)
        
        if audio_features is not None and audio_features.dim() == 3:
            audio_features = audio_features.unsqueeze(0)
        
        # Run inference
        try:
            # Generate transcription
            output = self.model.generate(
                audio=audio_features,
                video=video_features,
                num_beams=beam_size,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
            )
            
            # Decode output tokens to text
            transcription = self.model.decode_output(output)
            
            # Create result
            result = {
                "text": transcription,
                "tokens": output.tolist() if isinstance(output, torch.Tensor) else output,
                "metadata": {
                    "beam_size": beam_size,
                    "max_length": max_length,
                    "min_length": min_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            }
            
            # Add model config if available
            if hasattr(self.config, 'model'):
                if hasattr(self.config.model, '__dict__'):
                    result["metadata"]["model_config"] = self.config.model.__dict__
                else:
                    result["metadata"]["model_config"] = self.config.model
            
            return result
            
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise e
    
    def __call__(self, *args, **kwargs):
        """
        Call method for convenience
        """
        return self.transcribe(*args, **kwargs) 