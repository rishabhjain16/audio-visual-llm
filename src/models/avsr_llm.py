#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
import traceback

from .llm_module import LLMModule
from .av_hubert import AVHuBERTEncoder
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
        
        # Initialize state
        self.fusion_module = None
        self.audio_encoder = None
        self.video_encoder = None
        self.audio_projection = None
        self.video_projection = None
        
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
                    
                    # Create projection for audio encoder output
                    if hasattr(self, 'encoder_dim'):
                        self.audio_projection = nn.Linear(
                            self.audio_encoder.embedding_dim, 
                            self.encoder_dim
                        )
                        # Convert projection to float16 for memory efficiency
                        self.audio_projection.weight.data = self.audio_projection.weight.data.to(torch.float16)
                        if self.audio_projection.bias is not None:
                            self.audio_projection.bias.data = self.audio_projection.bias.data.to(torch.float16)
                        logging.info(f"Created audio projection from {self.audio_encoder.embedding_dim} to {self.encoder_dim}")
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
                if hasattr(config, "av_encoder_path") and config.av_encoder_path:
                    av_encoder_path = config.av_encoder_path
                    logging.info(f"Using AV-HuBERT model from: {av_encoder_path}")
                    
                    # Log the absolute path for debugging
                    abs_path = os.path.abspath(av_encoder_path) if not os.path.isabs(av_encoder_path) else av_encoder_path
                    logging.info(f"Absolute path to AV-HuBERT model: {abs_path}")
                    
                    # Check file permissions
                    if os.path.exists(abs_path):
                        try:
                            with open(abs_path, 'rb') as f:
                                # Just checking if we can read the file
                                pass
                            logging.info(f"Successfully verified read access to AV-HuBERT model file")
                        except PermissionError:
                            logging.error(f"Permission denied when accessing AV-HuBERT model file: {abs_path}")
                            logging.error(f"Please check file permissions")
                            raise
                    else:
                        # List files in the directory to help diagnose the issue
                        parent_dir = os.path.dirname(abs_path)
                        if os.path.exists(parent_dir):
                            files = os.listdir(parent_dir)
                            logging.error(f"AV-HuBERT model file not found: {abs_path}")
                            logging.error(f"Files in parent directory ({parent_dir}): {files[:10]}")
                            if len(files) > 10:
                                logging.error(f"... and {len(files) - 10} more files")
                        else:
                            logging.error(f"AV-HuBERT model directory not found: {parent_dir}")
                        raise FileNotFoundError(f"AV-HuBERT model file not found: {abs_path}")
                    
                    # Initialize AV-HuBERT for video
                    logging.info(f"Initializing AV-HuBERT encoder from {av_encoder_path}")
                    
                    # Create AVHuBERT encoder with output dimension matching LLM dimension
                    self.video_encoder = AVHuBERTEncoder(
                        checkpoint_path=av_encoder_path,
                        layer=getattr(config.model, "avhubert_layer", -1) if hasattr(config, 'model') else -1,
                        use_audio=False,
                        use_video=True,
                        freeze=getattr(config, "freeze_av_encoder", True),
                        finetune_layers=getattr(config, "finetune_avhubert_layers", []),
                        output_dim=self.llm_dim  # Ensure output dimension matches LLM
                    )
                    logging.info(f"Video encoder initialized with output dimension: {self.video_encoder.embedding_dim}")
                    
                    # Create projection for video encoder output
                    if hasattr(self, 'encoder_dim'):
                        self.video_projection = nn.Linear(
                            self.video_encoder.embedding_dim,
                            self.encoder_dim
                        )
                        # Convert projection to float16 for memory efficiency
                        self.video_projection.weight.data = self.video_projection.weight.data.to(torch.float16)
                        if self.video_projection.bias is not None:
                            self.video_projection.bias.data = self.video_projection.bias.data.to(torch.float16)
                        logging.info(f"Created video projection from {self.video_encoder.embedding_dim} to {self.encoder_dim}")
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
        
        # Initialize LLM module
        self.llm_module = None
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
            audio: Audio input tensor [batch_size, seq_len, feat_dim]
            video: Video input tensor [batch_size, seq_len, feat_dim]
            
        Returns:
            encoder_out: Encoded features
            encoder_padding_mask: Padding mask for encoded features
        """
        # Get max sequence length setting from config or use default
        max_seq_len = getattr(self.config, "max_seq_len", 250)  # Default to 250 if not specified
        
        # Get device and dtype for consistent tensor properties
        device = next(self.parameters()).device
        # Use float16 for memory efficiency and speed
        dtype = torch.float16
        
        # Initialize outputs
        audio_out = None
        video_out = None
        
        # Process audio if available
        if audio is not None and audio.size(0) > 0 and self.use_audio:
            if self.audio_encoder is None:
                logging.error("Audio encoder is not initialized! Make sure the whisper_path is correct.")
            else:
                try:
                    # Check audio dimensions
                    if len(audio.shape) != 2:
                        logging.error(f"Expected 2D audio [batch_size, time], got shape {audio.shape}")
                        # Reshape if possible
                        if len(audio.shape) == 3:  # [batch, time, features]
                            audio = audio.mean(dim=2)
                            logging.warning(f"Reshaped 3D audio to 2D: {audio.shape}")
                    
                    # Ensure audio is on the correct device and dtype
                    if audio.device != device or audio.dtype != dtype:
                        logging.info(f"Moving audio from {audio.device} (dtype: {audio.dtype}) to {device} (dtype: {dtype})")
                        audio = audio.to(device=device, dtype=dtype)
                    
                    # Log audio stats for debugging
                    logging.debug(f"Audio min: {audio.min().item()}, max: {audio.max().item()}, mean: {audio.mean().item()}")
                    if torch.isnan(audio).any():
                        logging.warning("Audio contains NaN values!")
                        # Replace NaNs with zeros
                        audio = torch.nan_to_num(audio, nan=0.0)
                    
                    # Encode audio
                    audio_out = self.audio_encoder(audio)
                    
                    if audio_out is None:
                        logging.warning("Audio encoder returned None. Using random features.")
                        # Create random features
                        batch_size = audio.size(0)
                        seq_len = 4
                        audio_out = torch.randn(
                            batch_size, seq_len, self.audio_encoder.embedding_dim,
                            device=device, dtype=dtype
                        )
                    else:
                        logging.info(f"Audio encoded successfully: shape={audio_out.shape}")
                    
                    # Ensure consistent dtype
                    if audio_out.dtype != dtype:
                        audio_out = audio_out.to(dtype=dtype)
                    
                    # Limit sequence length if needed for VRAM management
                    orig_seq_len = audio_out.size(1)
                    if orig_seq_len > max_seq_len:
                        logging.info(f"Truncating audio output from {orig_seq_len} to {max_seq_len}")
                        logging.info(f"  (To use more context, increase max_seq_len in your config)")
                        audio_out = audio_out[:, :max_seq_len, :]
                    
                    # Apply projection if needed
                    if hasattr(self, 'audio_projection') and self.audio_projection is not None:
                        try:
                            audio_out = self.audio_projection(audio_out)
                            logging.debug(f"Applied audio projection: {audio_out.shape}")
                        except Exception as e:
                            logging.error(f"Error in audio projection: {e}")
                            # Try to handle dimension mismatch
                            if "mat1 and mat2 shapes cannot be multiplied" in str(e) and hasattr(self, 'encoder_dim'):
                                # Create a new projection on the fly
                                logging.warning(f"Creating emergency projection from {audio_out.size(-1)} to {self.encoder_dim}")
                                emergency_proj = nn.Linear(audio_out.size(-1), self.encoder_dim, device=device)
                                audio_out = emergency_proj(audio_out)
                            else:
                                raise
                except Exception as e:
                    logging.error(f"Error processing audio: {e}")
                    logging.error(traceback.format_exc())
                    audio_out = None
        
        # Process video if available
        if video is not None and video.size(0) > 0 and self.use_video:
            if self.video_encoder is None:
                logging.error("Video encoder is not initialized! Make sure the av_encoder_path is correct.")
            else:
                try:
                    # Ensure video is on the correct device and dtype
                    if video.device != device or video.dtype != dtype:
                        logging.info(f"Moving video from {video.device} (dtype: {video.dtype}) to {device} (dtype: {dtype})")
                        video = video.to(device=device, dtype=dtype)
                    
                    # Encode video
                    video_out = self.video_encoder(video)
                    
                    # Check if video encoding was successful
                    if video_out is not None:
                        logging.info(f"Video encoded successfully: shape={video_out.shape}")
                        
                        # Ensure consistent dtype
                        if video_out.dtype != dtype:
                            video_out = video_out.to(dtype=dtype)
                        
                        # Limit sequence length if needed for VRAM management
                        orig_seq_len = video_out.size(1)
                        if orig_seq_len > max_seq_len:
                            logging.info(f"Truncating video output from {orig_seq_len} to {max_seq_len}")
                            logging.info(f"  (To use more context, increase max_seq_len in your config)")
                            video_out = video_out[:, :max_seq_len, :]
                        
                        # Apply projection if needed
                        if hasattr(self, 'video_projection') and self.video_projection is not None:
                            video_out = self.video_projection(video_out)
                            # Ensure consistent dtype after projection
                            if video_out.dtype != dtype:
                                video_out = video_out.to(dtype=dtype)
                        elif video_out is not None:
                            logging.error("Video projection is missing but video is encoded! Check model initialization.")
                    else:
                        logging.warning("Video encoder returned None for video_out")
                    
                except Exception as e:
                    logging.error(f"Error encoding video: {e}")
                    logging.error(traceback.format_exc())
                    video_out = None
        
        # Check if at least one modality was processed
        if audio_out is None and video_out is None:
            logging.warning("No valid inputs for encoder, returning None")
            return None, None
            
        # Combine features if both modalities are available
        if audio_out is not None and video_out is not None:
            # Ensure same batch size
            if audio_out.size(0) != video_out.size(0):
                logging.warning(f"Batch size mismatch: audio {audio_out.size(0)}, video {video_out.size(0)}")
                # Use the smallest batch size
                min_batch = min(audio_out.size(0), video_out.size(0))
                audio_out = audio_out[:min_batch]
                video_out = video_out[:min_batch]
            
            # Ensure same sequence length
            if audio_out.size(1) != video_out.size(1):
                # Use the smallest sequence length
                min_seq = min(audio_out.size(1), video_out.size(1))
                audio_out = audio_out[:, :min_seq, :]
                video_out = video_out[:, :min_seq, :]
            
            # Ensure consistent dtype
            if audio_out.dtype != dtype:
                audio_out = audio_out.to(dtype=dtype)
            if video_out.dtype != dtype:
                video_out = video_out.to(dtype=dtype)
            
            # Apply fusion module if available
            if self.fusion_module is not None:
                encoder_out = self.fusion_module(audio_out, video_out)
            else:
                # Simple concatenation along feature dimension
                encoder_out = torch.cat([audio_out, video_out], dim=2)
            
            # Ensure consistent dtype after fusion
            if encoder_out.dtype != dtype:
                encoder_out = encoder_out.to(dtype=dtype)
            
            # Create padding mask (all False if no padding)
            encoder_padding_mask = torch.zeros(encoder_out.size(0), encoder_out.size(1), dtype=torch.bool, device=encoder_out.device)
            
            return encoder_out, encoder_padding_mask
            
        # Use only audio if available
        elif audio_out is not None:
            # Ensure consistent dtype
            if audio_out.dtype != dtype:
                audio_out = audio_out.to(dtype=dtype)
            
            # Create padding mask (all False if no padding)
            encoder_padding_mask = torch.zeros(audio_out.size(0), audio_out.size(1), dtype=torch.bool, device=audio_out.device)
            
            return audio_out, encoder_padding_mask
            
        # Use only video if available
        elif video_out is not None:
            # Ensure consistent dtype
            if video_out.dtype != dtype:
                video_out = video_out.to(dtype=dtype)
            
            # Create padding mask (all False if no padding)
            encoder_padding_mask = torch.zeros(video_out.size(0), video_out.size(1), dtype=torch.bool, device=video_out.device)
            
            return video_out, encoder_padding_mask
            
        # Should never reach here due to earlier check
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
                if k not in llm_inputs:
                    llm_inputs[k] = v
            
            # Forward pass through LLM
            if self.llm_module is None:
                logging.error("LLM module is None! This could be due to initialization failure.")
                dummy_loss = torch.tensor(float('nan'), device=device, requires_grad=True)
                return {"loss": dummy_loss}
                
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