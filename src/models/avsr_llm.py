#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import WhisperModel, Wav2Vec2Model, HubertModel
from typing import Dict, List, Optional, Tuple, Any
import traceback
import time

from .llm_module import LLMModule
from ..avhubert_whisper.models.av_hubert import AVHuBERTEncoder
from .whisper_encoder import WhisperEncoder

class AVSRLLM(nn.Module):
    """
    Audio-Visual Speech Recognition with LLM
    """
    
    def __init__(
        self,
        config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the AVSR-LLM model
        
        Args:
            config: Model configuration
            device: Device to use
        """
        super().__init__()
        self.config = config
        self.device = device
        self.encoder_dim = getattr(config, "fusion_dim", 1024)
        self.use_audio = getattr(config, "use_audio", True)
        self.use_video = getattr(config, "use_video", True)
        self.debug = getattr(config, "debug", False)
        
        # Initialize components
        self.audio_encoder = None
        self.video_encoder = None
        self.fusion_module = None
        self.audio_projection = None
        self.video_projection = None
        self.llm_module = None
        
        # Get dimensions from config
        # Handle different config structures (direct attributes or nested under model)
        if hasattr(config, 'model'):
            self.audio_dim = getattr(config.model, "audio_dim", 80)
            self.video_dim = getattr(config.model, "video_dim", 512)  
            self.fusion_dim = getattr(config.model, "fusion_dim", 2048)
            self.llm_dim = getattr(config.model, "llm_dim", 2048)
        else:
            # Flat config structure
            self.audio_dim = getattr(config, "audio_dim", 80)
            self.video_dim = getattr(config, "video_dim", 512)
            self.fusion_dim = getattr(config, "fusion_dim", 2048)
            self.llm_dim = getattr(config, "llm_dim", 2048)
        
        # Ensure fusion_dim matches llm_dim to avoid dimension mismatch issues
        if self.fusion_dim != self.llm_dim:
            logging.warning(f"Fusion dimension ({self.fusion_dim}) doesn't match LLM dimension ({self.llm_dim}). Using LLM dimension for fusion.")
            self.fusion_dim = self.llm_dim
            
        # Store the encoder dimension for projections
        self.encoder_dim = self.fusion_dim
        
        # Initialize audio encoder
        if self.use_audio:
            try:
                logging.info("Initializing audio encoder...")
                
                # Check if whisper_path is provided
                if hasattr(config, "whisper_path") and config.whisper_path:
                    whisper_path = config.whisper_path
                    logging.info(f"Using Whisper model from: {whisper_path}")
                    
                    # If it's a local path, check if it exists
                    if whisper_path.startswith('/') or whisper_path.startswith('./'):
                        abs_path = os.path.abspath(whisper_path)
                        logging.info(f"Absolute path to Whisper model: {abs_path}")
                        
                        if not os.path.exists(abs_path):
                            # List files in the directory to help diagnose the issue
                            parent_dir = os.path.dirname(abs_path)
                            if os.path.exists(parent_dir):
                                files = os.listdir(parent_dir)
                                logging.error(f"Whisper model path not found: {abs_path}")
                                logging.error(f"Files in parent directory ({parent_dir}): {files[:10]}")
                                if len(files) > 10:
                                    logging.error(f"... and {len(files) - 10} more files")
                            else:
                                logging.error(f"Whisper model directory not found: {parent_dir}")
                            raise FileNotFoundError(f"Whisper model not found: {abs_path}")
                    else:
                        # Assume it's a Hugging Face model ID
                        logging.info(f"Using Whisper model from Hugging Face: {whisper_path}")
                        
                    # Initialize the encoder
                    self.audio_encoder = WhisperEncoder(
                        model_id=whisper_path,
                        freeze=getattr(config, "freeze_audio_encoder", True),
                        use_encoder_only=True
                    )
                    logging.info(f"Audio encoder initialized with output dimension: {self.audio_encoder.embedding_dim}")
                    
                    # Create projections if needed
                    if self.audio_encoder is not None and hasattr(self.audio_encoder, 'embedding_dim'):
                        audio_dim = self.audio_encoder.embedding_dim
                        if audio_dim != self.encoder_dim:
                            logging.info(f"Creating audio projection from {audio_dim} to {self.encoder_dim}")
                            self.audio_projection = nn.Linear(audio_dim, self.encoder_dim)
                            # Ensure it's float32
                            self.audio_projection = self.audio_projection.to(torch.float32)
                else:
                    logging.error("No whisper_path provided in config. This is required for audio processing!")
                    logging.error("Please set whisper_path in your config file")
                    raise ValueError("Missing whisper_path in config")
            except Exception as e:
                logging.error(f"Error initializing audio encoder: {e}")
                logging.error(traceback.format_exc())
                self.use_audio = False
        
        # Initialize video encoder
        if self.use_video:
            try:
                logging.info("Initializing video encoder...")
                
                # Check if av_encoder_path is provided
                av_encoder_path = getattr(config, "av_encoder_path", None)
                if av_encoder_path:
                    try:
                        from ..avhubert_whisper.models.av_hubert import AVHuBERTEncoder
                        
                        # Check if the model path exists
                        if not os.path.exists(av_encoder_path):
                            logging.error(f"AV-HuBERT model path does not exist: {av_encoder_path}")
                            raise FileNotFoundError(f"AV-HuBERT model not found at {av_encoder_path}")
                        
                        # Log absolute path for debugging
                        abs_path = os.path.abspath(av_encoder_path)
                        logging.info(f"Initializing AV-HuBERT encoder from: {abs_path}")
                        
                        # Initialize the encoder
                        self.video_encoder = AVHuBERTEncoder(
                            checkpoint_path=av_encoder_path,
                            use_audio=getattr(config, "use_avhubert_audio", False),  # Default to not using audio from AV-HuBERT
                            use_video=True,
                            freeze=getattr(config, "freeze_video_encoder", True),
                            output_dim=self.llm_dim
                        )
                        logging.info(f"Video encoder initialized with output dimension: {self.video_encoder.embedding_dim} (projected to {self.llm_dim})")
                        
                        # No need for additional projection since AVHuBERTEncoder handles projection internally
                        self.video_projection = None
                        
                        # Create projections if needed
                        if self.video_encoder is not None and hasattr(self.video_encoder, 'output_dim'):
                            video_dim = self.video_encoder.output_dim
                            if video_dim != self.encoder_dim:
                                logging.info(f"Creating video projection from {video_dim} to {self.encoder_dim}")
                                self.video_projection = nn.Linear(video_dim, self.encoder_dim)
                                # Ensure it's float32
                                self.video_projection = self.video_projection.to(torch.float32)
                    except Exception as e:
                        logging.error(f"Error initializing video encoder: {e}")
                        logging.error(traceback.format_exc())
                        self.use_video = False
                else:
                    logging.error("No av_encoder_path provided in config. This is required!")
                    logging.error("Please set av_encoder_path in your config file to the path of your AV-HuBERT model")
                    raise ValueError("Missing av_encoder_path in config")
            except Exception as e:
                logging.error(f"Error initializing video encoder: {e}")
                logging.error(traceback.format_exc())
                self.use_video = False
        
        # Check if at least one encoder is initialized
        if not self.use_audio and not self.use_video:
            logging.error("Both audio and video encoders failed to initialize! Model will not function correctly.")
        
        # Initialize fusion module
        if self.use_audio and self.use_video:
            try:
                from .fusion import SimpleFusion
                
                # Get encoder dimensions
                audio_dim = self.audio_encoder.embedding_dim if hasattr(self.audio_encoder, 'embedding_dim') else 1024
                video_dim = self.video_encoder.embedding_dim if hasattr(self.video_encoder, 'embedding_dim') else 1024
                
                logging.info(f"Using encoder dims: audio_dim={audio_dim}, video_dim={video_dim}, output_dim={self.encoder_dim}")
                
                # Create fusion module with fp16 support
                self.fusion_module = SimpleFusion(
                    audio_dim=audio_dim, 
                    video_dim=video_dim, 
                    output_dim=self.encoder_dim, 
                    use_fp16=True  # Use fp16 for better efficiency
                )
                
                logging.info(f"Fusion module initialized: {type(self.fusion_module).__name__}")
                
                # Freeze fusion if specified
                if getattr(config, "freeze_fusion", False):
                    for param in self.fusion_module.parameters():
                        param.requires_grad = False
                    logging.info("Fusion module is frozen")
                
            except Exception as e:
                logging.error(f"Error initializing fusion module: {e}")
                logging.error(traceback.format_exc())
                self.fusion_module = None
        else:
            logging.info("Only one modality is used, no fusion module needed")
            self.fusion_module = None
        
        # Initialize LLM module
        if hasattr(config, "llm_path") and config.llm_path:
            try:
                llm_path = config.llm_path
                logging.info(f"Initializing LLM module with model: {llm_path}")
                
                # Check if path exists (if it's a local path)
                if os.path.exists(llm_path) or not llm_path.startswith('/'):
                    try:
                        # List directory contents for debugging
                        if os.path.exists(llm_path) and os.path.isdir(llm_path):
                            logging.info(f"LLM directory contains: {os.listdir(llm_path)[:10]}")
                            
                        # Check for model files
                        model_files = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer.model']
                        for file in model_files:
                            file_path = os.path.join(llm_path, file)
                            if os.path.exists(file_path):
                                logging.info(f"Found model file: {file}")
                            else:
                                logging.warning(f"Missing model file: {file}")
                    except Exception as e:
                        logging.error(f"Error checking LLM directory: {e}")
                
                    # Initialize the LLM module with detailed error handling
                    try:
                        from .llm_module import LLMModule
                        self.llm_module = LLMModule(
                            model_name_or_path=llm_path,
                            encoder_dim=self.encoder_dim,
                            use_lora=getattr(config, "use_lora", True),
                            lora_r=getattr(config, "lora_r", 8),
                            lora_alpha=getattr(config, "lora_alpha", 16),
                            lora_dropout=getattr(config, "lora_dropout", 0.05),
                            prompt_template=getattr(config, "prompt_template", "Transcribe the speech: ")
                        )
                        logging.info("LLM module initialized successfully!")
                    except Exception as llm_error:
                        logging.error(f"Failed to initialize LLM module: {llm_error}")
                        logging.error(traceback.format_exc())
                        self.llm_module = None
                    
                    # Log LLM info
                    if self.llm_module is not None and hasattr(self.llm_module, "model") and hasattr(self.llm_module.model, "config"):
                        if hasattr(self.llm_module.model.config, "hidden_size"):
                            self.llm_dim = self.llm_module.model.config.hidden_size
                            logging.info(f"LLM initialized with hidden size: {self.llm_dim}")
                        else:
                            self.llm_dim = 2048  # Default fallback
                            logging.info(f"Using default hidden size: {self.llm_dim}")
                    else:
                        logging.warning("Could not determine LLM hidden size")
                else:
                    logging.error(f"LLM model path not found: {llm_path}")
            except Exception as e:
                logging.error(f"Error initializing LLM module: {e}")
                logging.error(traceback.format_exc())
                self.llm_module = None
        else:
            logging.error("No LLM path provided in config, model will not function correctly")
            
        # Check if LLM module is initialized
        if self.llm_module is None:
            logging.error("LLM module failed to initialize! Model will not function correctly.")
        
        # Move model to device - handle meta tensors correctly for PyTorch 2.6+
        try:
            # Try regular to() call first
            self.to(device)
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e):
                # For PyTorch 2.6+, use to_empty() for meta tensors
                logging.info(f"Using to_empty() for meta tensors (PyTorch 2.6+ compatibility)")
                self.to_empty(device=device)
                # Initialize parameters with random values
                for name, param in self.named_parameters():
                    if param.device.type == 'meta':
                        logging.warning(f"Parameter {name} is still on meta device, initializing with random values")
                        param.data = torch.randn_like(param.data, device=device)
            else:
                # Re-raise the exception if it's not about meta tensors
                raise
                
        logging.info(f"Initialized AVSR-LLM model on {device}")
        
        # Log model parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Model has {trainable_params:,} trainable parameters out of {total_params:,} total parameters")
    
    def encode(self, audio=None, video=None):
        """
        Encode audio and video inputs
        
        Args:
            audio: Audio tensor of shape [B, T]
            video: Video tensor of shape [B, T, H, W, C]
            
        Returns:
            Tuple of (encoder_output, padding_mask)
            encoder_output: Tensor of shape [B, T, D]
            padding_mask: Tensor of shape [B, T] - True for padded positions
        """
        try:
            # Get device
            device = next(self.parameters()).device
            
            # Process audio input
            audio_out = None
            if audio is not None and self.audio_encoder is not None:
                # Move audio to the correct device if necessary
                if audio.device != device:
                    logging.info(f"Moving audio from {audio.device} (dtype: {audio.dtype}) to {device} (dtype: {audio.dtype})")
                    audio = audio.to(device)
                
                # Convert audio to float32 for better compatibility
                if audio.dtype != torch.float32:
                    logging.info(f"Converting audio from {audio.dtype} to float32")
                    audio = audio.to(dtype=torch.float32)
                
                # Encode audio
                try:
                    audio_out = self.audio_encoder(audio)
                    
                    if audio_out is not None:
                        logging.info(f"Audio encoded successfully: shape={audio_out.shape}, dtype={audio_out.dtype}")
                        
                        # Convert to float32 if needed
                        if audio_out.dtype != torch.float32:
                            logging.info(f"Converting audio output from {audio_out.dtype} to float32")
                            audio_out = audio_out.to(torch.float32)
                        
                        # Project audio features if necessary
                        if hasattr(self, 'audio_projection') and self.audio_projection is not None and audio_out.size(-1) != self.encoder_dim:
                            logging.info(f"Applying audio projection: {audio_out.size(-1)} -> {self.encoder_dim}")
                            
                            # Ensure projection weights are float32
                            if self.audio_projection.weight.dtype != torch.float32:
                                logging.info(f"Converting audio_projection from {self.audio_projection.weight.dtype} to float32")
                                self.audio_projection = self.audio_projection.to(torch.float32)
                                
                            audio_out = self.audio_projection(audio_out)
                            logging.debug(f"Audio features projected: shape={audio_out.shape}")
                except Exception as e:
                    logging.error(f"Error encoding audio: {e}")
                    logging.error(traceback.format_exc())
                    audio_out = None
            
            # Process video input
            video_out = None
            if video is not None and self.video_encoder is not None:
                # Move video to the correct device if necessary
                if video.device != device:
                    logging.info(f"Moving video from {video.device} (dtype: {video.dtype}) to {device} (dtype: {video.dtype})")
                    video = video.to(device)
                
                logging.info(f"Video input shape: {video.shape}")
                
                # Convert video to float32 for better compatibility
                if video.dtype != torch.float32:
                    logging.info(f"Converting video from {video.dtype} to float32")
                    video = video.to(dtype=torch.float32)
                
                # Encode video
                try:
                    video_out = self.video_encoder(video)
                    logging.info(f"Video encoded successfully: shape={video_out.shape}, dtype={video_out.dtype}")
                    
                    # Convert to float32 if needed
                    if video_out.dtype != torch.float32:
                        logging.info(f"Converting video output from {video_out.dtype} to float32")
                        video_out = video_out.to(torch.float32)
                    
                    # Project video features if necessary
                    if hasattr(self, 'video_projection') and self.video_projection is not None and video_out.size(-1) != self.encoder_dim:
                        logging.info(f"Applying video projection: {video_out.size(-1)} -> {self.encoder_dim}")
                        video_out = self.video_projection(video_out)
                        logging.debug(f"Video features projected: shape={video_out.shape}")
                except Exception as e:
                    logging.error(f"Error encoding video: {e}")
                    logging.error(traceback.format_exc())
                    video_out = None
            
            # Check if at least one modality was processed
            if audio_out is None and video_out is None:
                logging.warning("No valid inputs for encoder, returning None, None")
                return None, None
            
            # Return the appropriate output based on available modalities
            if audio_out is not None and video_out is not None:
                # Multimodal case - both audio and video available
                
                # Ensure both are float32
                if audio_out.dtype != torch.float32:
                    logging.info(f"Converting audio output from {audio_out.dtype} to float32")
                    audio_out = audio_out.to(torch.float32)
                
                if video_out.dtype != torch.float32:
                    logging.info(f"Converting video output from {video_out.dtype} to float32")
                    video_out = video_out.to(torch.float32)
                
                # Check sequence length mismatch
                if audio_out.size(1) != video_out.size(1):
                    logging.warning(f"Sequence length mismatch: audio {audio_out.size(1)}, video {video_out.size(1)}")
                    min_seq = min(audio_out.size(1), video_out.size(1))
                    audio_out = audio_out[:, :min_seq, :]
                    video_out = video_out[:, :min_seq, :]
                
                # Apply fusion module if available
                if self.fusion_module is not None:
                    try:
                        encoder_out = self.fusion_module(audio_out, video_out)
                        logging.info(f"Applied fusion module: output shape {encoder_out.shape}")
                    except Exception as e:
                        logging.error(f"Error in fusion module: {e}")
                        logging.error(traceback.format_exc())
                        # Fall back to simple concatenation
                        logging.warning("Falling back to simple concatenation due to fusion error")
                        try:
                            # Ensure both tensors have the same data type (float32)
                            if audio_out.dtype != torch.float32:
                                audio_out = audio_out.to(torch.float32)
                            if video_out.dtype != torch.float32:
                                video_out = video_out.to(torch.float32)
                                
                            concat_out = torch.cat([audio_out, video_out], dim=-1)
                            logging.info(f"Concatenated features: shape={concat_out.shape}, dtype={concat_out.dtype}")
                            
                            # Project to expected dimension if needed
                            if concat_out.size(-1) != self.encoder_dim:
                                if not hasattr(self, 'concat_proj') or self.concat_proj is None:
                                    logging.info(f"Creating emergency concat projection from {concat_out.size(-1)} to {self.encoder_dim}")
                                    self.concat_proj = nn.Linear(concat_out.size(-1), self.encoder_dim).to(device=concat_out.device, dtype=torch.float32)
                                
                                # Ensure projection is float32
                                if self.concat_proj.weight.dtype != torch.float32:
                                    logging.info(f"Converting concat_proj from {self.concat_proj.weight.dtype} to float32")
                                    self.concat_proj = self.concat_proj.to(torch.float32)
                                
                                encoder_out = self.concat_proj(concat_out)
                            else:
                                encoder_out = concat_out
                        except Exception as e:
                            logging.error(f"Error in encode method: {e}")
                            logging.error(traceback.format_exc())
                            # Fall back to audio only as a last resort
                            logging.warning("Concatenation failed, using audio features only")
                            encoder_out = audio_out
                else:
                    # No fusion module, just concatenate and project if needed
                    logging.warning("No fusion module defined, using simple concatenation")
                    concat_out = torch.cat([audio_out, video_out], dim=-1)
                    # Project to expected dimension if needed
                    if concat_out.size(-1) != self.encoder_dim:
                        if not hasattr(self, 'concat_proj'):
                            logging.info(f"Creating emergency concat projection from {concat_out.size(-1)} to {self.encoder_dim}")
                            self.concat_proj = nn.Linear(concat_out.size(-1), self.encoder_dim, device=concat_out.device)
                        encoder_out = self.concat_proj(concat_out)
                    else:
                        encoder_out = concat_out
                
                # Create padding mask (all False if no padding)
                encoder_padding_mask = torch.zeros(encoder_out.size(0), encoder_out.size(1), dtype=torch.bool, device=encoder_out.device)
                
                return encoder_out, encoder_padding_mask
            
            # Single modality case - only audio or only video
            elif audio_out is not None:
                # Only audio available
                logging.info("Using audio modality only")
                
                # Create padding mask (all False if no padding)
                encoder_padding_mask = torch.zeros(audio_out.size(0), audio_out.size(1), dtype=torch.bool, device=audio_out.device)
                
                return audio_out, encoder_padding_mask
            
            # Only video available
            elif video_out is not None:
                logging.info("Using video modality only")
                
                # Create padding mask (all False if no padding)
                encoder_padding_mask = torch.zeros(video_out.size(0), video_out.size(1), dtype=torch.bool, device=video_out.device)
                
                return video_out, encoder_padding_mask
            
            # Fallback - should never reach here if checks above are correct
            logging.error("Unexpected condition in encode method - no valid outputs but didn't return None, None")
            return None, None
            
        except Exception as e:
            logging.error(f"Error in encode method: {e}")
            logging.error(traceback.format_exc())
            return None, None
    
    def forward(
        self,
        audio=None,
        video=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        return_loss=True,
        **kwargs
    ):
        """
        Forward pass through the model
        
        Args:
            audio: Audio features [batch_size, seq_len, dim]
            video: Video features [batch_size, seq_len, dim]
            inputs_embeds: Pre-encoded features [batch_size, seq_len, dim] (if provided, audio/video are ignored)
            attention_mask: Attention mask [batch_size, seq_len] (0 for masked positions, 1 for unmasked)
            labels: Token ids for training [batch_size, seq_len]
            return_loss: Whether to return loss
            **kwargs: Additional arguments for the LLM
            
        Returns:
            Dictionary containing the loss and outputs if return_loss=True,
            otherwise just the outputs
        """
        try:
            # Get device
            device = next(self.parameters()).device
            
            # Move all inputs to the correct device
            if audio is not None and audio.device != device:
                audio = audio.to(device)
            if video is not None and video.device != device:
                video = video.to(device)
            if inputs_embeds is not None and inputs_embeds.device != device:
                inputs_embeds = inputs_embeds.to(device)
            if attention_mask is not None and attention_mask.device != device:
                attention_mask = attention_mask.to(device)
            if labels is not None and isinstance(labels, torch.Tensor) and labels.device != device:
                labels = labels.to(device)
            
            # Encode audio/video if provided
            if inputs_embeds is None:
                encoder_out, encoder_padding_mask = self.encode(audio, video)
            else:
                # Use provided inputs_embeds
                encoder_out = inputs_embeds
                # Create padding mask from attention_mask if provided
                if attention_mask is not None:
                    encoder_padding_mask = ~attention_mask.bool()
                else:
                    # If no attention mask, assume all tokens are valid
                    encoder_padding_mask = torch.zeros(
                        encoder_out.size(0), encoder_out.size(1),
                        dtype=torch.bool, device=device
                    )
            
            # Handle case where encoder returns None
            if encoder_out is None or encoder_out.size(0) == 0:
                logging.warning("No encoder output, returning dummy loss")
                return {"loss": torch.tensor(float('nan'), device=device)}
                
            # Handle labels processing for loss calculation
            processed_labels = None
            if labels is not None and return_loss:
                # Convert labels to tensor if it's a list
                if isinstance(labels, list):
                    # Try to convert text labels to token ids using the tokenizer
                    if self.llm_module and hasattr(self.llm_module, 'tokenizer'):
                        try:
                            tokenized = self.llm_module.tokenizer(labels, return_tensors='pt', padding=True)
                            processed_labels = tokenized['input_ids'].to(device)
                            logging.info(f"Tokenized text labels to tensor: {processed_labels.shape}")
                        except Exception as e:
                            logging.error(f"Error tokenizing labels: {e}")
                            processed_labels = None
                else:
                    # Already a tensor
                    processed_labels = labels
            
            # For training stability, use batch size 1
            batch_size = 1
            
            # Prepare inputs for LLM
            llm_inputs = {
                'inputs_embeds': encoder_out[:batch_size],
                'attention_mask': (~encoder_padding_mask[:batch_size]).float() if encoder_padding_mask is not None else None,
            }
            
            # Add labels if processed successfully
            if processed_labels is not None and return_loss:
                llm_inputs['labels'] = processed_labels[:batch_size]
                logging.debug(f"Added labels with shape {processed_labels[:batch_size].shape} for loss calculation")
            
            # Add any additional kwargs
            for k, v in kwargs.items():
                if k not in llm_inputs and k != 'debug':  # Filter out 'debug' parameter
                    llm_inputs[k] = v
            
            # Forward pass through LLM
            if self.llm_module is None:
                logging.error("LLM module is None! This could be due to initialization failure.")
                dummy_loss = torch.tensor(float('nan'), device=device, requires_grad=True)
                # Return a dictionary with dummy values to avoid breaking the training loop
                return {
                    "loss": dummy_loss,
                    "logits": torch.randn(1, 1, 1000, device=device),  # Dummy logits
                    "input_shape": encoder_out.shape if encoder_out is not None else (1, 1, self.fusion_dim),
                    "dummy_output": True  # Flag to indicate this is a dummy output
                }
                
            outputs = self.llm_module(**llm_inputs)
            
            return outputs
            
        except Exception as e:
            logging.error(f"Error in forward: {e}")
            logging.error(traceback.format_exc())
            
            # Return dummy output on error
            return {"loss": torch.tensor(float('nan'), device=next(self.parameters()).device)}
    
    @torch.no_grad()
    def generate(
        self,
        audio=None,
        video=None,
        padding_mask=None,
        prompt=None,
        max_length=None,
        min_length=None,
        num_beams=None,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        length_penalty=None,
    ):
        """
        Generate text from audio/video features
        
        Args:
            audio: Audio features tensor
            video: Video features tensor
            padding_mask: Padding mask where True indicates padding
            prompt: Text prompt to condition generation
            max_length: Maximum generation length
            min_length: Minimum generation length
            num_beams: Beam size for beam search
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            length_penalty: Length penalty
            
        Returns:
            Generated token IDs
        """
        # Encode audio/video features
        encoded, padding_mask = self.encode(audio, video)
        
        # Default values from config
        if max_length is None:
            max_length = getattr(self.config, "max_length", 256)
        if num_beams is None:
            num_beams = getattr(self.config, "num_beams", 5)
        
        # Add prompt if specified
        input_ids = None
        if prompt is not None:
            # Use the specified prompt
            prompt_text = prompt
        elif self.prompt_template is not None:
            # Use the default prompt template
            prompt_text = self.prompt_template
        else:
            prompt_text = None
            
        if prompt_text is not None:
            input_ids = self.llm.tokenizer.encode(prompt_text, return_tensors="pt").to(encoded.device)
        
        # Generate text
        outputs = self.llm.generate(
            encoder_hidden_states=encoded,
            encoder_attention_mask=None if padding_mask is None else ~padding_mask,
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty
        )
        
        return outputs
    
    def decode_output(self, token_ids):
        """
        Decode output token IDs to text
        
        Args:
            token_ids: Output token IDs
            
        Returns:
            Decoded text
        """
        return self.llm.tokenizer.decode(token_ids[0], skip_special_tokens=True)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None):
        """
        Load model from a checkpoint file
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to use (defaults to CUDA if available)
            
        Returns:
            model: Loaded model
        """
        try:
            logging.info(f"Loading model from checkpoint: {checkpoint_path}")
            
            # Set device if not provided
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract config
            config = checkpoint.get('config', None)
            if config is None:
                logging.warning("Checkpoint does not contain config, using default config")
                from types import SimpleNamespace
                config = SimpleNamespace()
            
            # Create model with the same config
            model = cls(config=config, device=device)
            
            # Load state dict
            state_dict = checkpoint.get('model_state_dict', None)
            if state_dict is not None:
                model.load_state_dict(state_dict)
                logging.info("Model state loaded from checkpoint")
            else:
                logging.warning("No model state dictionary found in checkpoint")
            
            # Set additional attributes from checkpoint
            if 'epoch' in checkpoint:
                model.current_epoch = checkpoint['epoch']
                logging.info(f"Loaded from epoch {model.current_epoch}")
            
            logging.info(f"Model loaded successfully from checkpoint on {device}")
            return model
            
        except Exception as e:
            logging.error(f"Error loading model from checkpoint: {e}")
            logging.error(traceback.format_exc())
            raise