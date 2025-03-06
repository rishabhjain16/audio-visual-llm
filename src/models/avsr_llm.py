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
                    
                    # Initialize the encoder
                    self.video_encoder = AVHuBERTEncoder(
                        checkpoint_path=av_encoder_path,
                        layer=getattr(config, "avhubert_layer", -1),
                        use_audio=False,  # We're using Whisper for audio
                        use_video=True,
                        freeze=getattr(config, "freeze_av_encoder", True),
                        finetune_layers=getattr(config, "finetune_avhubert_layers", [])
                    )
                    logging.info(f"Video encoder initialized with output dimension: {self.video_encoder.embedding_dim}")
                    
                    # Create projection for video encoder output
                    if hasattr(self, 'encoder_dim'):
                        self.video_projection = nn.Linear(
                            self.video_encoder.embedding_dim,
                            self.encoder_dim
                        )
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
                    self.llm_module = LLMModule(
                        model_name_or_path=llm_path,
                        encoder_dim=self.encoder_dim,
                        use_lora=getattr(config, "use_lora", True),
                        lora_r=getattr(config, "lora_r", 8),
                        lora_alpha=getattr(config, "lora_alpha", 16),
                        lora_dropout=getattr(config, "lora_dropout", 0.05),
                        prompt_template=getattr(config, "prompt_template", "Transcribe the speech: ")
                    )
                    
                    # Log LLM info
                    if hasattr(self.llm_module, "model") and hasattr(self.llm_module.model, "config"):
                        if hasattr(self.llm_module.model.config, "hidden_size"):
                            self.llm_hidden_size = self.llm_module.model.config.hidden_size
                            logging.info(f"LLM initialized with hidden size: {self.llm_hidden_size}")
                        else:
                            self.llm_hidden_size = 2048  # Default fallback
                            logging.info(f"Using default hidden size: {self.llm_hidden_size}")
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
                    # Ensure audio is on the correct device and dtype
                    if audio.device != device or audio.dtype != dtype:
                        logging.info(f"Moving audio from {audio.device} (dtype: {audio.dtype}) to {device} (dtype: {dtype})")
                        audio = audio.to(device=device, dtype=dtype)
                    
                    # Encode audio
                    audio_out = self.audio_encoder(audio)
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
                        audio_out = self.audio_projection(audio_out)
                        # Ensure consistent dtype after projection
                        if audio_out.dtype != dtype:
                            audio_out = audio_out.to(dtype=dtype)
                    elif audio_out is not None:
                        logging.error("Audio projection is missing but audio is encoded! Check model initialization.")
                    
                except Exception as e:
                    logging.error(f"Error encoding audio: {e}")
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
            # Log GPU info if debug is enabled
            if self.debug and torch.cuda.is_available():
                device_idx = 0 if isinstance(self.device, str) else self.device.index
                free_mem, total_mem = torch.cuda.mem_get_info(device_idx)
                free_mem = free_mem / (1024 ** 3)  # Convert to GB
                total_mem = total_mem / (1024 ** 3)  # Convert to GB
                used_mem = total_mem - free_mem
                logging.debug(f"GPU memory before forward: {used_mem:.2f}GB used / {total_mem:.2f}GB total")
            
            # Get device
            device = next(self.parameters()).device
            
            # Check if inputs are pre-encoded or need encoding
            encoder_out = None
            encoder_padding_mask = None
            
            if inputs_embeds is not None:
                # Use pre-encoded inputs
                logging.debug("Using pre-encoded inputs")
                encoder_out = inputs_embeds
                
                # Convert attention mask to padding mask (True for padding, False for content)
                if attention_mask is not None:
                    encoder_padding_mask = ~(attention_mask.bool())
                else:
                    # Create all-False padding mask if not provided
                    encoder_padding_mask = torch.zeros(inputs_embeds.size(0), inputs_embeds.size(1), 
                                                      dtype=torch.bool, device=inputs_embeds.device)
            else:
                # Encode audio and video
                logging.debug("Encoding audio and video inputs")
                if self.debug:
                    if audio is not None:
                        logging.debug(f"Audio input shape: {audio.shape}, device: {audio.device}, dtype: {audio.dtype}")
                    else:
                        logging.debug("Audio input is None")
                        
                    if video is not None:
                        logging.debug(f"Video input shape: {video.shape}, device: {video.device}, dtype: {video.dtype}")
                    else:
                        logging.debug("Video input is None")
                
                # Encode audio and video
                encoder_out, encoder_padding_mask = self.encode(audio, video)
                
                # Log encoder output for debugging
                if self.debug:
                    if encoder_out is not None:
                        logging.debug(f"Encoder output shape: {encoder_out.shape}, device: {encoder_out.device}")
                    else:
                        logging.warning("Encoder returned None for encoder_out")
                        
                    if encoder_padding_mask is not None:
                        logging.debug(f"Encoder padding mask shape: {encoder_padding_mask.shape}, device: {encoder_padding_mask.device}")
                    else:
                        logging.debug("Encoder padding mask is None")
            
            # Ensure all inputs are on the same device
            if encoder_out is not None and encoder_out.device != device:
                logging.debug(f"Moving encoder_out from {encoder_out.device} to {device}")
                encoder_out = encoder_out.to(device)
            if encoder_padding_mask is not None and encoder_padding_mask.device != device:
                logging.debug(f"Moving encoder_padding_mask from {encoder_padding_mask.device} to {device}")
                encoder_padding_mask = encoder_padding_mask.to(device)
            if labels is not None and hasattr(labels, 'device') and labels.device != device:
                logging.debug(f"Moving labels from {labels.device} to {device}")
                labels = labels.to(device)
            
            # Check if encoder returned valid output
            if encoder_out is None or encoder_out.size(0) == 0:
                logging.warning("Encoder returned empty output, generating random features")
                batch_size = 1
                if labels is not None:
                    batch_size = labels.size(0)
                encoder_dim = self.encoder_dim
                encoder_out = torch.randn(batch_size, 50, encoder_dim, device=device)
                encoder_padding_mask = torch.zeros(batch_size, 50, dtype=torch.bool, device=device)
            
            # Pass encoded features to LLM
            if self.llm_module is not None:
                # Prepare inputs for the LLM module
                llm_inputs = {
                    'inputs_embeds': encoder_out,  # Changed from encoder_out to inputs_embeds
                    'attention_mask': (~encoder_padding_mask).float() if encoder_padding_mask is not None else None,
                }
                
                # Add labels if provided and return_loss is True
                if labels is not None and return_loss:
                    llm_inputs['labels'] = labels
                
                # Add any additional kwargs
                for k, v in kwargs.items():
                    llm_inputs[k] = v
                
                # Call the LLM module with the prepared inputs
                outputs = self.llm_module(**llm_inputs)
                
                # Log LLM output for debugging
                if self.debug:
                    if isinstance(outputs, dict):
                        for k, v in outputs.items():
                            if isinstance(v, torch.Tensor):
                                logging.debug(f"LLM output[{k}] shape: {v.shape}, device: {v.device}")
                    else:
                        logging.debug(f"LLM output type: {type(outputs)}")
                
                # Convert tuple/list to dict if needed
                if not isinstance(outputs, dict) and return_loss:
                    if isinstance(outputs, tuple) and len(outputs) > 0:
                        outputs = {"loss": outputs[0], "logits": outputs[1] if len(outputs) > 1 else None}
                    elif hasattr(outputs, "loss"):
                        outputs = {"loss": outputs.loss, "logits": outputs.logits if hasattr(outputs, "logits") else None}
                
                # Log GPU info after forward pass if debug is enabled
                if self.debug and torch.cuda.is_available():
                    device_idx = 0 if isinstance(self.device, str) else self.device.index
                    free_mem, total_mem = torch.cuda.mem_get_info(device_idx)
                    free_mem = free_mem / (1024 ** 3)  # Convert to GB
                    total_mem = total_mem / (1024 ** 3)  # Convert to GB
                    used_mem = total_mem - free_mem
                    logging.debug(f"GPU memory after forward: {used_mem:.2f}GB used / {total_mem:.2f}GB total")
                
                return outputs
            else:
                logging.error("LLM module is None, cannot process inputs")
                return {"loss": torch.tensor(float('nan'), device=device)}
                
        except Exception as e:
            logging.error(f"Error in forward: {e}")
            logging.error(traceback.format_exc())
            
            # Return dummy output in case of error
            device = self.device
            dummy_loss = torch.tensor(float('nan'), device=device)
            dummy_dict = {"loss": dummy_loss}
            return dummy_dict
    
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