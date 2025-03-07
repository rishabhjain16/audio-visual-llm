#!/usr/bin/env python3
"""
AVHuBERT-Whisper model that combines AVHuBERT (video) and Whisper (audio) encoders
with a language model for audio-visual speech recognition.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, LlamaModel, LlamaForCausalLM
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import traceback

from .av_hubert import AVHuBERTEncoder

class AVHuBERTWhisperModel(nn.Module):
    """
    A model combining AVHuBERT for video encoding, Whisper for audio encoding,
    and a language model for transcription.
    """
    
    def __init__(
        self,
        llm_path: str = "meta-llama/Llama-2-7b-chat-hf",
        whisper_model: str = "openai/whisper-medium",
        avhubert_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = False,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_encoders: bool = True,
        freeze_llm: bool = False,
        modality: str = "both",
        max_seq_len: int = 256,
        fusion_scale: float = 0.5,
    ):
        """
        Initialize the AVHuBERT-Whisper model
        
        Args:
            llm_path: Path to the LLM model (Llama)
            whisper_model: Name or path of Whisper model
            avhubert_path: Path to the AVHuBERT checkpoint
            device: Device to use for inference
            use_fp16: Whether to use mixed precision
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout rate
            freeze_encoders: Whether to freeze encoders
            freeze_llm: Whether to freeze the LLM
            modality: Which modalities to use: "audio", "video", or "both"
            max_seq_len: Maximum sequence length for encoder output
            fusion_scale: Weight for audio in fusion (0.5 = equal weight)
        """
        super().__init__()
        
        # Save parameters
        self.device = device
        self.use_fp16 = use_fp16
        self.freeze_encoders = freeze_encoders
        self.freeze_llm = freeze_llm
        self.modality = modality
        self.max_seq_len = max_seq_len
        self.fusion_scale = fusion_scale
        
        # Initialize all sub-models
        self.llm = self._load_llm(llm_path, use_lora, lora_r, lora_alpha, lora_dropout, freeze_llm)
        self.audio_encoder = self._load_whisper_model(whisper_model, freeze_encoders)
        self.video_encoder = self._load_avhubert_model(avhubert_path, freeze_encoders)
        
        # Get LLM input dimension
        self.llm_dim = self._get_llm_dim()
        
        # Create projections for encoder outputs if needed
        self._setup_projections()
        
        # Log the model architecture
        self._log_model_architecture()
        
    def _load_llm(self, llm_path, use_lora, lora_r, lora_alpha, lora_dropout, freeze_llm):
        """Load the language model"""
        logging.info(f"Loading LLM from {llm_path}")
        
        try:
            # Load LLM for causal language modeling
            llm = LlamaForCausalLM.from_pretrained(
                llm_path,
                device_map=self.device if self.device != "cpu" else None,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32
            )
            
            # Apply LoRA if requested
            if use_lora and not freeze_llm:
                from peft import get_peft_model, LoraConfig, TaskType
                
                logging.info(f"Applying LoRA to LLM with rank={lora_r}, alpha={lora_alpha}")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=freeze_llm,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
                )
                llm = get_peft_model(llm, peft_config)
                logging.info("LoRA applied successfully")
            
            # Freeze LLM if requested
            if freeze_llm:
                logging.info("Freezing LLM")
                for param in llm.parameters():
                    param.requires_grad = False
            
            return llm
            
        except Exception as e:
            logging.error(f"Error loading LLM: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def _load_whisper_model(self, whisper_model, freeze):
        """Load Whisper model for audio encoding"""
        logging.info(f"Loading Whisper model from {whisper_model}")
        
        try:
            # Load Whisper model for encoding audio
            model = WhisperModel.from_pretrained(
                whisper_model,
                device_map=self.device if self.device != "cpu" else None,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32
            )
            
            # Freeze if requested
            if freeze:
                logging.info("Freezing Whisper model")
                for param in model.parameters():
                    param.requires_grad = False
            
            return model
            
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def _load_avhubert_model(self, avhubert_path, freeze):
        """Load AVHuBERT model for video encoding"""
        if avhubert_path is None:
            logging.warning("No AVHuBERT path provided, skipping video encoder initialization")
            return None
            
        logging.info(f"Loading AVHuBERT from {avhubert_path}")
        
        try:
            # Load AVHuBERT model for encoding video
            model = AVHuBERTEncoder(
                checkpoint_path=avhubert_path,
                use_audio=False,
                use_video=True,
                freeze=freeze,
                output_dim=self.llm_dim if hasattr(self, "llm_dim") else 1024
            )
            
            return model
            
        except Exception as e:
            logging.error(f"Error loading AVHuBERT model: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def _get_llm_dim(self):
        """Get the input dimension of the LLM"""
        # Get LLM input embedding dimension
        if hasattr(self.llm, "get_input_embeddings"):
            llm_dim = self.llm.get_input_embeddings().weight.shape[1]
            logging.info(f"Detected LLM input dimension: {llm_dim}")
            return llm_dim
        else:
            logging.warning("Could not detect LLM input dimension, using default 4096")
            return 4096
    
    def _setup_projections(self):
        """Set up projections from encoder outputs to LLM input dimension"""
        # Whisper projections
        if self.audio_encoder is not None:
            whisper_dim = self.audio_encoder.config.d_model
            logging.info(f"Setting up projection from Whisper ({whisper_dim}) to LLM ({self.llm_dim})")
            self.audio_projection = nn.Linear(whisper_dim, self.llm_dim)
        else:
            self.audio_projection = None
        
        # AVHuBERT projections (if needed)
        if self.video_encoder is not None:
            # Check if AVHuBERT already has output_dim set to match LLM
            if self.video_encoder.output_dim != self.llm_dim:
                logging.info(f"Setting up projection from AVHuBERT ({self.video_encoder.embedding_dim}) to LLM ({self.llm_dim})")
                self.video_projection = nn.Linear(self.video_encoder.embedding_dim, self.llm_dim)
            else:
                logging.info(f"AVHuBERT already produces outputs matching LLM dimension ({self.llm_dim})")
                self.video_projection = None
        else:
            self.video_projection = None
    
    def _log_model_architecture(self):
        """Log the model architecture for debugging"""
        logging.info("=" * 50)
        logging.info("AVHuBERT-Whisper Model Architecture")
        logging.info("=" * 50)
        logging.info(f"Modality: {self.modality}")
        logging.info(f"LLM dimension: {self.llm_dim}")
        
        if self.audio_encoder is not None:
            logging.info(f"Audio: Whisper {self.audio_encoder.config.model_type} ({self.audio_encoder.config.d_model} dim)")
        
        if self.video_encoder is not None:
            logging.info(f"Video: AVHuBERT ({self.video_encoder.embedding_dim} dim)")
        
        logging.info(f"Fusion scale: {self.fusion_scale} (audio weight)")
        logging.info("=" * 50)
    
    def _pad_or_truncate(self, x, target_len):
        """Pad or truncate sequence to target length"""
        if x is None:
            return None
            
        current_len = x.size(1)
        if current_len > target_len:
            # Truncate
            return x[:, :target_len, :]
        elif current_len < target_len:
            # Pad with zeros
            padding = torch.zeros(
                (x.size(0), target_len - current_len, x.size(2)),
                dtype=x.dtype, device=x.device
            )
            return torch.cat([x, padding], dim=1)
        return x

    def forward(
        self,
        audio=None,
        video=None,
        text=None,
        prompt=None,
        labels=None,
        return_loss=True,
        **kwargs
    ):
        """Forward pass through the model"""
        try:
            # Log the modality that's being used
            logging.info(f"Forward pass using modality: {self.modality}")
            
            # Validate inputs based on modality
            if self.modality == "audio" and audio is None:
                raise ValueError("Audio modality selected but no audio input provided")
            elif self.modality == "video" and video is None:
                raise ValueError("Video modality selected but no video input provided")
            elif self.modality == "both" and (audio is None or video is None):
                raise ValueError("Both modalities selected but inputs are missing")
            
            # Get batch size
            if audio is not None:
                batch_size = audio.size(0)
            elif video is not None:
                batch_size = video.size(0)
            else:
                raise ValueError("Neither audio nor video inputs provided")
                
            # Log input shapes
            if audio is not None:
                logging.debug(f"Audio input shape: {audio.shape}")
            if video is not None:
                logging.debug(f"Video input shape: {video.shape}")
            
            # Encode audio and/or video
            audio_features = None
            video_features = None
            
            # Audio encoding (with Whisper)
            if (self.modality == "audio" or self.modality == "both") and audio is not None:
                audio_features = self.encode_audio(audio)
                logging.debug(f"Audio features shape: {audio_features.shape}")
            
            # Video encoding (with AVHuBERT)
            if (self.modality == "video" or self.modality == "both") and video is not None:
                video_features = self.encode_video(video)
                logging.debug(f"Video features shape: {video_features.shape}")
            
            # Combine features based on modality
            if self.modality == "audio":
                encoder_out = audio_features
            elif self.modality == "video":
                encoder_out = video_features
            else:  # both modalities
                # Ensure both features have the same sequence length
                max_len = max(
                    audio_features.size(1) if audio_features is not None else 0,
                    video_features.size(1) if video_features is not None else 0
                )
                
                # Pad or truncate to match
                audio_features = self._pad_or_truncate(audio_features, max_len)
                video_features = self._pad_or_truncate(video_features, max_len)
                
                # Weighted sum of features
                encoder_out = (
                    self.fusion_scale * audio_features +
                    (1 - self.fusion_scale) * video_features
                )
            
            # Ensure we respect max_seq_len
            if encoder_out.size(1) > self.max_seq_len:
                encoder_out = encoder_out[:, :self.max_seq_len, :]
                logging.debug(f"Truncated encoder output to {self.max_seq_len}")
            
            # Store sequence length for masking
            seq_len = encoder_out.size(1)
            
            # Ensure encoder output matches LLM's expected dtype
            llm_input_dtype = next(self.llm.parameters()).dtype
            if encoder_out.dtype != llm_input_dtype:
                logging.info(f"Converting encoder output from {encoder_out.dtype} to {llm_input_dtype}")
                encoder_out = encoder_out.to(llm_input_dtype)
            
            # Create initial attention mask (all 1s)
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)
                
            # Handle prompt if provided
            if prompt is not None:
                logging.info(f"Processing prompt with shape: {prompt.shape}")
                embedding_layer = self.llm.get_input_embeddings()
                prompt_embeds = embedding_layer(prompt.to(self.device))
                logging.info(f"Prompt embeddings shape: {prompt_embeds.shape}, Encoder output shape: {encoder_out.shape}")
                
                # Ensure batch dimension matches
                if prompt_embeds.size(0) != encoder_out.size(0):
                    logging.warning(f"Batch size mismatch: prompt_embeds={prompt_embeds.size(0)}, encoder_out={encoder_out.size(0)}")
                    # Expand prompt if needed
                    if prompt_embeds.size(0) == 1 and encoder_out.size(0) > 1:
                        prompt_embeds = prompt_embeds.expand(encoder_out.size(0), -1, -1)
                
                # Concatenate along sequence dimension (dim=1)
                encoder_out = torch.cat([prompt_embeds, encoder_out], dim=1)
                logging.info(f"Combined output shape after adding prompt: {encoder_out.shape}")
                
                # Update attention mask to cover prompt tokens as well
                attention_mask = torch.ones((batch_size, encoder_out.size(1)), dtype=torch.long, device=self.device)
                logging.info(f"Updated attention mask shape: {attention_mask.shape}")
            else:
                logging.info(f"No prompt provided, using encoder output directly with shape: {encoder_out.shape}")
                logging.info(f"Attention mask shape: {attention_mask.shape}")
                
            # Prepare labels for loss computation if needed
            if return_loss and labels is not None:
                # Adjust labels to match sequence length
                if labels.size(1) != encoder_out.size(1):
                    logging.warning(f"Label sequence length ({labels.size(1)}) doesn't match encoder output ({encoder_out.size(1)})")
                    
                # Forward pass with labels for loss computation
                outputs = self.llm(
                    inputs_embeds=encoder_out,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
            else:
                # Forward pass without loss computation
                outputs = self.llm(
                    inputs_embeds=encoder_out,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            return outputs
        
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def encode_audio(self, audio, attention_mask=None):
        """Encode audio with Whisper"""
        if self.audio_encoder is None:
            raise ValueError("Audio encoder not initialized")
        
        with torch.set_grad_enabled(not self.freeze_encoders):
            # Whisper encoding
            audio_outputs = self.audio_encoder(
                audio,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get last hidden states (features)
            features = audio_outputs.last_hidden_state
            
            # Apply projection to match LLM dimension
            if self.audio_projection is not None:
                features = self.audio_projection(features)
                
        return features
    
    def encode_video(self, video, attention_mask=None):
        """Encode video with AVHuBERT"""
        if self.video_encoder is None:
            raise ValueError("Video encoder not initialized")
        
        with torch.set_grad_enabled(not self.freeze_encoders):
            # AVHuBERT encoding
            features = self.video_encoder(video, padding_mask=attention_mask)
            
            # Apply projection if needed (although AVHuBERT should already handle this)
            if self.video_projection is not None:
                features = self.video_projection(features)
                
        return features
    
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
        """Generate text based on audio/video inputs"""
        # Set default generation parameters
        if max_length is None:
            max_length = 200
        if num_beams is None:
            num_beams = 5
            
        # Process audio/video encodings
        with torch.no_grad():
            # Encode audio and/or video first
            audio_features = None
            video_features = None
            
            # Audio encoding
            if (self.modality == "audio" or self.modality == "both") and audio is not None:
                audio_features = self.encode_audio(audio)
                
            # Video encoding
            if (self.modality == "video" or self.modality == "both") and video is not None:
                video_features = self.encode_video(video, padding_mask)
                
            # Combine features based on modality
            if self.modality == "audio":
                encoder_out = audio_features
            elif self.modality == "video":
                encoder_out = video_features
            else:  # both modalities
                # Ensure both features have the same sequence length
                max_len = max(
                    audio_features.size(1) if audio_features is not None else 0,
                    video_features.size(1) if video_features is not None else 0
                )
                
                # Pad or truncate to match
                audio_features = self._pad_or_truncate(audio_features, max_len)
                video_features = self._pad_or_truncate(video_features, max_len)
                
                # Weighted sum of features
                encoder_out = (
                    self.fusion_scale * audio_features +
                    (1 - self.fusion_scale) * video_features
                )
            
            # Get batch size
            batch_size = encoder_out.size(0)
            seq_len = encoder_out.size(1)
            
            # Create attention mask (all 1s)
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)
            
            # Handle prompt if provided
            if prompt is not None:
                embedding_layer = self.llm.get_input_embeddings()
                prompt_embeds = embedding_layer(prompt.to(self.device))
                
                # Concatenate along sequence dimension (dim=1)
                encoder_out = torch.cat([prompt_embeds, encoder_out], dim=1)
                
                # Update attention mask
                attention_mask = torch.ones((batch_size, encoder_out.size(1)), dtype=torch.long, device=self.device)
                
            # Generate text with the LLM
            generation_outputs = self.llm.generate(
                inputs_embeds=encoder_out,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length if min_length is not None else 0,
                num_beams=num_beams,
                temperature=temperature if temperature is not None else 1.0,
                top_p=top_p if top_p is not None else 1.0,
                top_k=top_k if top_k is not None else 50,
                repetition_penalty=repetition_penalty if repetition_penalty is not None else 1.0,
                length_penalty=length_penalty if length_penalty is not None else 1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            # Process generated token IDs
            return {
                "generated_ids": generation_outputs.sequences,
                "scores": generation_outputs.scores,
            }
    
    def save_pretrained(self, output_dir):
        """Save the model to the output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LLM
        llm_path = os.path.join(output_dir, "llm")
        os.makedirs(llm_path, exist_ok=True)
        self.llm.save_pretrained(llm_path)
        
        # Save configurable parameters
        import json
        config = {
            "modality": self.modality,
            "max_seq_len": self.max_seq_len,
            "fusion_scale": self.fusion_scale,
            "freeze_encoders": self.freeze_encoders,
            "freeze_llm": self.freeze_llm,
            "use_fp16": self.use_fp16,
        }
        
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f)
            
        logging.info(f"Model saved to {output_dir}")
        
        # Save state dictionary for loading outside of HF ecosystem
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
    @classmethod
    def from_pretrained(cls, model_dir):
        """Load the model from a pretrained directory"""
        import json
        
        # Load config
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
            
        # Initialize model with config
        model = cls(
            llm_path=os.path.join(model_dir, "llm"),
            modality=config.get("modality", "both"),
            max_seq_len=config.get("max_seq_len", 256),
            fusion_scale=config.get("fusion_scale", 0.5),
            freeze_encoders=config.get("freeze_encoders", True),
            freeze_llm=config.get("freeze_llm", False),
            use_fp16=config.get("use_fp16", False),
        )
        
        # Load state dictionary if exists
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            
        return model 