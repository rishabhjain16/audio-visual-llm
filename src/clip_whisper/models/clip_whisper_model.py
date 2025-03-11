import os
import torch
import torch.nn as nn
import logging
import traceback
import json
from typing import Optional, Tuple, Union, Dict, Any
from transformers import (
    WhisperModel,
    WhisperProcessor,
    CLIPVisionModel,
    CLIPProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    BitsAndBytesConfig
)
import datetime
import gc
from peft import LoraConfig, get_peft_model
from .modality_connector import ModalityConnector
import time

class ClipWhisperModel(nn.Module):
    """
    An Audio-Visual Speech Recognition model that combines CLIP and Whisper with an LLM.
    Uses Whisper for audio, CLIP for video, and an LLM (e.g., Llama) for language modeling.
    
    Supports three modality modes:
    - audio: Uses only the Whisper encoder and audio connector
    - video: Uses only the CLIP encoder and video connector
    - both: Uses both encoders and fuses the features
    """
    
    def _get_gpu_memory_usage(self):
        """Get current GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0}
            
        return {
            "allocated": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved": torch.cuda.memory_reserved() / (1024 * 1024)
        }
    
    def _log_memory_usage(self, stage):
        """Log memory usage at a specific stage"""
        if not torch.cuda.is_available():
            return
            
        memory = self._get_gpu_memory_usage()
        logging.info(f"GPU Memory [{stage}] - Allocated: {memory['allocated']:.2f} MB, Reserved: {memory['reserved']:.2f} MB")
    
    def _track_component_memory(self, component_name, callback):
        """Track memory usage of a specific component"""
        if not torch.cuda.is_available():
            initial = {"allocated": 0, "reserved": 0}
            result = callback()
            final = {"allocated": 0, "reserved": 0}
            diff = {"allocated": 0, "reserved": 0}
        else:
            # Clear cache and collect garbage to get more accurate measurements
            torch.cuda.empty_cache()
            gc.collect()
            
            # Measure initial memory
            initial = self._get_gpu_memory_usage()
            
            # Execute the callback (load model component)
            result = callback()
            
            # Measure final memory
            final = self._get_gpu_memory_usage()
            
            # Calculate difference
            diff = {
                "allocated": final["allocated"] - initial["allocated"],
                "reserved": final["reserved"] - initial["reserved"]
            }
        
        # Log memory usage
        logging.info(f"GPU Memory for {component_name}: +{diff['allocated']:.2f} MB allocated, " 
                    f"+{diff['reserved']:.2f} MB reserved")
        
        return result
    
    def __init__(
        self,
        llm_path: str = "meta-llama/Llama-2-7b-chat-hf",
        whisper_model: str = "openai/whisper-medium",
        clip_model: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = False,
        use_4bit: bool = False,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_encoders: bool = True,
        freeze_llm: bool = False,
        modality: str = "both",
        max_seq_len: int = 256,
        fusion_scale: float = 0.5,
        connector_type: str = "simple",
        _provided_tokenizer=None,
    ):
        """
        Initialize the ClipWhisper model
        
        Args:
            llm_path: Path to the LLM model (Llama)
            whisper_model: Name or path of Whisper model
            clip_model: Name or path of CLIP model
            device: Device to use for inference
            use_fp16: Whether to use mixed precision (FP16)
            use_4bit: Whether to use 4-bit quantization for the LLM
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout rate
            freeze_encoders: Whether to freeze encoders
            freeze_llm: Whether to freeze the LLM
            modality: Which modalities to use: "audio", "video", or "both"
            max_seq_len: Maximum sequence length for encoders
            fusion_scale: Scaling factor for audio-video fusion (0-1)
            connector_type: Type of connector to use. Options:
                - "simple": Basic linear projection
                - "deep": Multi-layer projection with residual connections
                - "conv": Convolutional projection for sequence patterns
                - "attention": Self-attention projection for long sequences
                - "adaptive": Adaptive projection based on sequence length
                - "cross_modal": Cross-modal attention between audio and video
                - "qformer": Query-based transformer for multimodal fusion
                - "perceiver": Perceiver-IO architecture for efficient multimodal processing
            _provided_tokenizer: Optional pre-loaded tokenizer
        """
        super().__init__()
        
        # Save parameters
        self.device = device
        self.use_fp16 = use_fp16
        self.use_4bit = use_4bit
        self.freeze_encoders = freeze_encoders
        self.freeze_llm = freeze_llm
        self.modality = modality
        self.max_seq_len = max_seq_len
        self.fusion_scale = fusion_scale
        self.connector_type = connector_type
        self._provided_tokenizer = _provided_tokenizer  # Store the provided tokenizer
        self.log_param_updates = False  # Initialize this to ensure it's always defined
        
        # Set modality with clear logging
        modality_banner = "=" * 80
        logging.info(f"\n{modality_banner}")
        logging.info(f"INITIALIZING CLIP-WHISPER MODEL WITH MODALITY: {self.modality.upper()}")
        logging.info(f"{modality_banner}")
        
        # Store configuration options
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Initialize memory tracking
        self.memory_stats = {
            'components': {},
            'total_initial': self._get_gpu_memory_usage(),
        }
        
        # Log initial GPU memory
        self._log_memory_usage("Initial")
        
        # Track base memory usage before loading any models
        base_memory = self._get_gpu_memory_usage()["allocated"]
        
        # Always load the LLM regardless of modality
        self.llm, self.tokenizer = self._track_component_memory("LLM",
                    lambda: self._load_llm(llm_path, use_lora, lora_r, lora_alpha, lora_dropout, freeze_llm, use_4bit))
        
        # Get LLM memory footprint
        llm_memory = self._get_gpu_memory_usage()["allocated"] - base_memory
        logging.info(f"LLM memory usage: {llm_memory:.2f} MB")
        
        # Load Whisper if needed for audio
        if self.modality in ["audio", "both"]:
            whisper_base_memory = self._get_gpu_memory_usage()["allocated"]
            self.whisper, self.whisper_processor = self._track_component_memory("Whisper",
                            lambda: self._load_whisper_model(whisper_model, freeze_encoders))
            whisper_memory = self._get_gpu_memory_usage()["allocated"] - whisper_base_memory
            logging.info(f"Whisper model memory usage: {whisper_memory:.2f} MB")
            
            self.audio_dim = self.whisper.config.d_model  # Whisper dimension
            logging.info(f"Loaded Whisper model for {self.modality} modality")
        else:
            # Create placeholders for audio components when not used
            self.whisper = None
            self.whisper_processor = None
            # Use a default dimension that matches Whisper for compatibility
            self.audio_dim = 1024  # Standard Whisper dimension
            logging.info(f"Whisper model not loaded (not needed for {self.modality} modality)")
        
        # Load CLIP if needed for video
        if self.modality in ["video", "both"]:
            clip_base_memory = self._get_gpu_memory_usage()["allocated"]
            self.clip, self.clip_processor = self._track_component_memory("CLIP",
                        lambda: self._load_clip_model(clip_model, freeze_encoders))
            clip_memory = self._get_gpu_memory_usage()["allocated"] - clip_base_memory
            logging.info(f"CLIP model memory usage: {clip_memory:.2f} MB")
            
            self.video_dim = self.clip.config.hidden_size  # CLIP dimension
            logging.info(f"Loaded CLIP model for {self.modality} modality")
        else:
            # Create placeholders for video components when not used
            self.clip = None
            self.clip_processor = None
            # Use a default dimension that matches CLIP for compatibility
            self.video_dim = 768  # Standard CLIP dimension
            logging.info(f"CLIP model not loaded (not needed for {self.modality} modality)")
        
        # Get LLM dimension
        self.llm_dim = self._get_llm_dim()
        
        # Create projections for encoder outputs
        self._setup_projections()
        
        # Log the model architecture
        self._log_model_architecture()
        
        # Move model to the specified device
        self.to(device)
        
        # Log final GPU memory after loading all components
        self._log_memory_usage("Final (after loading all components)")
        
        # Calculate total memory usage
        self.memory_stats['total_final'] = self._get_gpu_memory_usage()
        self.memory_stats['total_used'] = {
            "allocated": self.memory_stats['total_final']["allocated"] - self.memory_stats['total_initial']["allocated"],
            "reserved": self.memory_stats['total_final']["reserved"] - self.memory_stats['total_initial']["reserved"]
        }
        
        # Log total memory usage
        logging.info(f"Total GPU memory used by model: {self.memory_stats['total_used']['allocated']:.2f} MB allocated")
        
        # Log the model configuration
        logging.info("Model initialization complete")
        logging.info(f"Model configuration: modality={modality}, "
                   f"freeze_encoders={freeze_encoders}, "
                   f"freeze_llm={freeze_llm}, "
                   f"use_fp16={use_fp16}, "
                   f"use_lora={use_lora}, "
                   f"lora_r={lora_r}, "
                   f"lora_alpha={lora_alpha}, "
                   f"lora_dropout={lora_dropout}, "
                   f"max_seq_len={max_seq_len}, "
                   f"fusion_scale={fusion_scale}")
        
        # Store initialization parameters for saving
        self.init_params = {
            "llm_path": llm_path,
            "whisper_model": whisper_model,
            "clip_model": clip_model,
            "use_fp16": use_fp16,
            "use_lora": use_lora,
            "use_4bit": use_4bit,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "freeze_encoders": freeze_encoders,
            "freeze_llm": freeze_llm,
            "modality": modality,
            "max_seq_len": max_seq_len,
            "fusion_scale": fusion_scale,
            "connector_type": connector_type,
        }
        
        # Count and log trainable parameters by component
        self._log_parameter_count()

        # Initialize batch index tracking
        self._current_batch_idx = 0

    def _pad_or_truncate(self, features, target_len):
        """Pad or truncate features to target length"""
        if features is None:
            return None
        
        current_len = features.size(1)
        
        # Return as is if already matching
        if current_len == target_len:
            return features
            
        # Truncate if longer
        if current_len > target_len:
            return features[:, :target_len, :]
            
        # Pad if shorter
        padding_len = target_len - current_len
        padding = torch.zeros(
            (features.size(0), padding_len, features.size(2)),
            dtype=features.dtype,
            device=features.device
        )
        return torch.cat([features, padding], dim=1)

    def _track_sequence_length(self, seq_len):
        """Track sequence length statistics for monitoring"""
        if not hasattr(self, '_seq_len_stats'):
            self._seq_len_stats = {
                'count': 0,
                'min': float('inf'),
                'max': 0,
                'sum': 0,
                'lengths': []
            }
        
        # Update stats
        self._seq_len_stats['count'] += 1
        self._seq_len_stats['min'] = min(self._seq_len_stats['min'], seq_len)
        self._seq_len_stats['max'] = max(self._seq_len_stats['max'], seq_len)
        self._seq_len_stats['sum'] += seq_len
        
        # Keep only the last 1000 lengths to avoid memory issues
        self._seq_len_stats['lengths'].append(seq_len)
        if len(self._seq_len_stats['lengths']) > 1000:
            self._seq_len_stats['lengths'] = self._seq_len_stats['lengths'][-1000:]
        
        # Log stats every 100 samples
        if self._seq_len_stats['count'] % 100 == 0:
            avg = self._seq_len_stats['sum'] / self._seq_len_stats['count']
            recent_avg = sum(self._seq_len_stats['lengths']) / len(self._seq_len_stats['lengths'])
            logging.info(f"Encoder sequence length stats - Min: {self._seq_len_stats['min']}, "
                        f"Max: {self._seq_len_stats['max']}, "
                        f"Avg (all): {avg:.2f}, "
                        f"Avg (recent): {recent_avg:.2f}")

    def encode(self, audio=None, video=None, prompt=None):
        """Encode audio, video, and prompt into embeddings."""
        # Encode audio if available
        audio_features = None
        if self.modality in ["audio", "both"] and audio is not None:
            audio_features, _ = self.encode_audio(audio)  # Unpack the tuple
        
        # Encode video if available
        video_features = None
        if self.modality in ["video", "both"] and video is not None:
            video_features = self.encode_video(video)
        
        # Fuse features if both modalities are used
        if audio_features is not None and video_features is not None:
            max_len = min(self.max_seq_len, max(audio_features.size(1), video_features.size(1)))
            audio_features = self._pad_or_truncate(audio_features, max_len)
            video_features = self._pad_or_truncate(video_features, max_len)
            encoder_output = self.fusion_scale * audio_features + (1 - self.fusion_scale) * video_features
        elif audio_features is not None:
            encoder_output = audio_features
        elif video_features is not None:
            encoder_output = video_features
        else:
            raise ValueError("Both audio and video inputs cannot be None")
        
        # Handle prompt if provided
        if prompt is not None:
            prompt_embeds = self._embed_prompt(prompt)
            encoder_output = torch.cat([prompt_embeds, encoder_output], dim=1)
        
        # Ensure encoder output matches LLM's expected dtype
        llm_input_dtype = next(self.llm.parameters()).dtype
        if encoder_output.dtype != llm_input_dtype:
            encoder_output = encoder_output.to(llm_input_dtype)
        
        # Create attention mask
        attention_mask = torch.ones((encoder_output.size(0), encoder_output.size(1)), dtype=torch.long, device=self.device)
        
        return encoder_output, attention_mask

    def _embed_prompt(self, prompt):
        if prompt is None:
            return None
        
        # Ensure prompt is not too long
        max_prompt_len = 32
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len
            ).input_ids.to(self.device)
        else:
            prompt_ids = prompt.to(self.device)
            if prompt_ids.size(1) > max_prompt_len:
                prompt_ids = prompt_ids[:, :max_prompt_len]
        
        # Get embeddings
        embedding_layer = self.llm.get_input_embeddings()
        prompt_embeds = embedding_layer(prompt_ids)
        
        return prompt_embeds

    def forward(self, audio=None, video=None, prompt=None, labels=None, return_loss=True):
        """Forward pass for the model."""
        if self.use_fp16:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Initialize batch index if not exists
                if not hasattr(self, '_current_batch_idx'):
                    self._current_batch_idx = 0
                
                # Only log audio conversion once
                if self._current_batch_idx == 0 and audio is not None:
                    logging.info(f"Audio input dtype: {audio.dtype} -> float16")
                
                # Reduce logging frequency
                if self._current_batch_idx % 100 == 0:
                    if audio is not None:
                        logging.debug(f"[Batch {self._current_batch_idx}] Audio shape: {audio.shape}")
                
                encoder_output, attention_mask = self.encode(audio, video, prompt)
                
                if self._current_batch_idx % 100 == 0:
                    logging.debug(f"[Batch {self._current_batch_idx}] Features shape: {encoder_output.shape}")
                
                # Update batch index
                self._current_batch_idx += 1
                
                # Remove the fallback to text
                if labels is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    
                    # Ensure labels match the sequence length of encoder output
                    if labels.size(1) != encoder_output.size(1):
                        # Truncate or pad labels to match encoder output length
                        target_length = encoder_output.size(1)
                        if labels.size(1) > target_length:
                            labels = labels[:, :target_length]
                        else:
                            padding = torch.full(
                                (labels.size(0), target_length - labels.size(1)),
                                self.tokenizer.pad_token_id,
                                device=labels.device
                            )
                            labels = torch.cat([labels, padding], dim=1)
                
                # Run LLM on encoder output
                if return_loss and labels is not None:
                    outputs = self.llm(
                        inputs_embeds=encoder_output,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True
                    )
                else:
                    outputs = self.llm(
                        inputs_embeds=encoder_output,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                
                # Return loss and logits
                if return_loss:
                    return {"loss": outputs.loss, "logits": outputs.logits}
                else:
                    return {"logits": outputs.logits}
        else:
            # Encode inputs
            encoder_output, attention_mask = self.encode(audio, video, prompt)
            
            # Mask padding tokens in labels (if any)
            if labels is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Run LLM on encoder output
            if return_loss and labels is not None:
                outputs = self.llm(
                    inputs_embeds=encoder_output,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
            else:
                outputs = self.llm(
                    inputs_embeds=encoder_output,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            # Return loss and logits
            if return_loss:
                return {"loss": outputs.loss, "logits": outputs.logits}
            else:
                return {"logits": outputs.logits}

        if labels is not None:
            assert encoder_output.size(1) == labels.size(1), "Encoder output and labels must have same sequence length"

    def save_pretrained(self, output_dir):
        """Save model to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the projectors with consistent naming
        logging.info(f"Saving connectors to {output_dir}")
        try:
            torch.save(self.audio_connector.state_dict(), os.path.join(output_dir, "audio_connector.pt"))
            torch.save(self.video_connector.state_dict(), os.path.join(output_dir, "video_connector.pt"))
            
            torch.save({
                "modality": self.modality,
                "max_seq_len": self.max_seq_len,
                "fusion_scale": self.fusion_scale,
                "freeze_encoders": self.freeze_encoders,
                "freeze_llm": self.freeze_llm,
                "audio_dim": self.audio_dim,
                "video_dim": self.video_dim,
                "llm_dim": self.llm_dim,
                "connector_type": self.connector_type
            }, os.path.join(output_dir, "config.pt"))
        except Exception as e:
            logging.error(f"Error saving connectors: {e}")
        
        # Save tokenizer and config
        logging.info(f"Saving tokenizer to {output_dir}")
        try:
            self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
            
            # Save configuration
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(self.init_params, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving tokenizer and config: {e}")
            
        # Save the LLM
        llm_dir = os.path.join(output_dir, "llm")
        logging.info(f"Saving LLM to {llm_dir}")
        try:
            self.llm.save_pretrained(llm_dir)
        except Exception as e:
            logging.error(f"Error saving LLM: {e}")
            logging.error(traceback.format_exc())
            
        # Save the encoders if needed
        if not self.freeze_encoders:
            try:
                whisper_dir = os.path.join(output_dir, "whisper")
                logging.info(f"Saving Whisper to {whisper_dir}")
                self.whisper.save_pretrained(whisper_dir)
                
                clip_dir = os.path.join(output_dir, "clip")
                logging.info(f"Saving CLIP to {clip_dir}")
                self.clip.save_pretrained(clip_dir)
            except Exception as e:
                logging.error(f"Error saving encoders: {e}")
        
        # Verify the save
        self._verify_save(output_dir)
        
        logging.info(f"Model saved to {output_dir}")
            
    def _verify_save(self, output_dir):
        """Verify that all necessary files were saved"""
        expected_files = [
            "audio_connector.pt",
            "video_connector.pt",
            "model_config.json",
            "config.json",
            os.path.join("tokenizer", "tokenizer_config.json"),
            os.path.join("llm", "config.json"),
        ]
        
        missing_files = []
        for file in expected_files:
            if not os.path.exists(os.path.join(output_dir, file)):
                missing_files.append(file)
        
        if missing_files:
            logging.warning(f"Missing files in saved model: {missing_files}")
        else:
            logging.info("All expected files saved successfully")
    
    @classmethod
    def from_pretrained(cls, model_dir, tokenizer=None):
        """Load model from saved checkpoint"""
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist")
        
        # Load model configuration
        config_path = os.path.join(model_dir, "model_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Model config {config_path} does not exist")
        
        with open(config_path, "rb") as f:
            config = torch.load(f)
        
        # If tokenizer is provided, we'll use it instead of loading from llm_path
        llm_path = os.path.join(model_dir, "llm")
        
        # Create model with the saved configuration
        model = cls(
            llm_path=llm_path,
            whisper_model=os.path.join(model_dir, "whisper"),
            clip_model=os.path.join(model_dir, "clip"),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            use_fp16=False,  # Default to stable settings when loading
            use_lora=True,
            freeze_encoders=config.get("freeze_encoders", True),
            freeze_llm=config.get("freeze_llm", False),
            modality=config.get("modality", "both"),
            max_seq_len=config.get("max_seq_len", 256),
            fusion_scale=config.get("fusion_scale", 0.5),
            connector_type=config.get("connector_type", "simple"),
            _provided_tokenizer=tokenizer,  # Pass the tokenizer to __init__
        )
        
        # Load the projectors
        model.audio_connector.load_state_dict(torch.load(os.path.join(model_dir, "audio_connector.pt")))
        model.video_connector.load_state_dict(torch.load(os.path.join(model_dir, "video_connector.pt")))
        
        logging.info(f"Model loaded from {model_dir}")
        model._verify_save(model_dir)
        
        return model
    
    def _load_whisper_model(self, whisper_model, freeze_encoders):
        """Load the Whisper model for audio encoding"""
        logging.info(f"Loading Whisper model: {whisper_model}")
        compute_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        try:
            processor = WhisperProcessor.from_pretrained(whisper_model)
            model = WhisperModel.from_pretrained(whisper_model, torch_dtype=compute_dtype)
            
            if freeze_encoders:
                self._freeze_model(model)
            
            # Log the model's dtype
            logging.info(f"Whisper model dtype: {next(model.parameters()).dtype}")
            
            return model, processor
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            raise

    def _load_clip_model(self, clip_model, freeze_encoders):
        """Load the CLIP model for visual encoding"""
        logging.info(f"Loading CLIP model: {clip_model}")
        compute_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        try:
            processor = CLIPProcessor.from_pretrained(clip_model)
            model = CLIPVisionModel.from_pretrained(clip_model, torch_dtype=compute_dtype)
            
            if freeze_encoders:
                self._freeze_model(model)
            
            # Log the model's dtype
            logging.info(f"CLIP model dtype: {next(model.parameters()).dtype}")
            
            return model, processor
        except Exception as e:
            logging.error(f"Error loading CLIP model: {e}")
            raise

    def _freeze_model(self, model):
        """Freeze a model's parameters"""
        for param in model.parameters():
            param.requires_grad = False

    def _load_llm(self, llm_path, use_lora, lora_r, lora_alpha, lora_dropout, freeze_llm, use_4bit):
        """Load the language model"""
        logging.info(f"Loading LLM from {llm_path}")
        
        if use_4bit:
            try:
                logging.info("Using 4-bit quantization for LLM")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if self.use_fp16 else torch.float32,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                llm = AutoModelForCausalLM.from_pretrained(
                    llm_path,
                    device_map="auto",
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if self.use_fp16 else torch.float32
                )
            except Exception as e:
                logging.error(f"Error loading LLM with 4-bit quantization: {e}")
                raise
        else:
            # Standard loading without quantization
            llm = AutoModelForCausalLM.from_pretrained(llm_path, device_map={"": 0}, torch_dtype=torch.float16 if self.use_fp16 else torch.float32)
        
        # Use provided tokenizer if available, otherwise load from llm_path
        if self._provided_tokenizer is not None:
            logging.info("Using provided tokenizer")
            tokenizer = self._provided_tokenizer
        else:
            logging.info(f"Loading tokenizer from {llm_path}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(llm_path)
            except Exception as e:
                logging.error(f"Error loading tokenizer from {llm_path}: {e}")
                # If loading from llm_path fails, try loading from tokenizer subdirectory
                tokenizer_path = os.path.join(os.path.dirname(llm_path), "tokenizer")
                if os.path.exists(tokenizer_path):
                    logging.info(f"Attempting to load tokenizer from {tokenizer_path}")
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                else:
                    raise
        
        # Ensure tokenizer has a pad token
        if tokenizer.pad_token is None:
            logging.info("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # Add the pad token to the vocabulary
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        
        # Apply LoRA if requested
        if use_lora:
            logging.info(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            
            # Configure target modules based on model type
            if 'llama' in llm_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                # Default target modules for other model types
                target_modules = ["query", "key", "value", "dense"]
                
            # Create LoRA config with stable initialization settings
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
                init_lora_weights="gaussian",  # Use gaussian initialization for better stability
                fan_in_fan_out=False,  # Better for most models
            )
            
            # Apply LoRA to model
            try:
                llm = get_peft_model(llm, peft_config)
                logging.info("LoRA applied successfully")
                
                # Scale down the initial LoRA weights to prevent gradient explosions
                with torch.no_grad():
                    scaling_factor = 0.01  # Helps prevent initial gradient explosions
                    param_count = 0
                    
                    for name, param in llm.named_parameters():
                        if 'lora_' in name and param.requires_grad:
                            # Apply a small scaling factor to LoRA weights
                            param.data = param.data * scaling_factor
                            param_count += 1
                    
                    logging.info(f"Scaled down initial values for {param_count} LoRA parameters by factor {scaling_factor}")
                
            except Exception as e:
                logging.error(f"Error applying LoRA: {e}")
                logging.warning("Continuing without LoRA")
                # Continue without LoRA rather than failing entirely
        
        # Freeze LLM weights if requested
        if freeze_llm:
            logging.info("Freezing LLM weights (only LoRA weights will be trained)")
            for param in llm.parameters():
                param.requires_grad = False
                
            # If using LoRA, ensure LoRA weights are trainable
            if use_lora:
                for n, p in llm.named_parameters():
                    if 'lora' in n:
                        p.requires_grad = True
                
        return llm, tokenizer

    def _log_parameter_count(self):
        """Log the number of parameters in each component"""
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        # Count parameters by component
        llm_params = sum(p.numel() for p in self.llm.parameters())
        whisper_params = sum(p.numel() for p in self.whisper.parameters()) if self.whisper is not None else 0
        clip_params = sum(p.numel() for p in self.clip.parameters()) if self.clip is not None else 0
        
        # Count connector parameters
        connector_params = 0
        if hasattr(self, 'audio_connector') and self.audio_connector is not None:
            connector_params += sum(p.numel() for p in self.audio_connector.parameters())
        if hasattr(self, 'video_connector') and self.video_connector is not None:
            connector_params += sum(p.numel() for p in self.video_connector.parameters())
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Log parameter counts
        logging.info("\n" + "=" * 80)
        logging.info("MODEL PARAMETER COUNTS")
        logging.info("=" * 80)
        logging.info(f"{'Component':<20} {'Parameters':<15} {'% of Total':<15}")
        logging.info("-" * 80)
        
        # Format numbers with commas for readability
        def fmt_count(count):
            return f"{count:,}"
        
        logging.info(f"{'LLM':<20} {fmt_count(llm_params):<15} {llm_params/total_params*100:.2f}%")
        
        if self.whisper is not None:
            logging.info(f"{'Whisper':<20} {fmt_count(whisper_params):<15} {whisper_params/total_params*100:.2f}%")
        
        if self.clip is not None:
            logging.info(f"{'CLIP':<20} {fmt_count(clip_params):<15} {clip_params/total_params*100:.2f}%")
        
        logging.info(f"{'Connectors':<20} {fmt_count(connector_params):<15} {connector_params/total_params*100:.2f}%")
        logging.info(f"{'TOTAL':<20} {fmt_count(total_params):<15} 100.00%")
        logging.info("-" * 80)
        logging.info(f"{'Trainable':<20} {fmt_count(trainable_params):<15} {trainable_params/total_params*100:.2f}%")
        logging.info(f"{'Frozen':<20} {fmt_count(total_params-trainable_params):<15} {(total_params-trainable_params)/total_params*100:.2f}%")
        logging.info("=" * 80 + "\n")
    
    def encode_audio(self, audio, attention_mask=None):
        """Encode audio input using Whisper"""
        if self.whisper is None:
            # For compatibility, return a dummy tensor with the right shape
            logging.warning("Audio encoding attempted but Whisper model not loaded - returning dummy tensor")
            return torch.zeros((audio.size(0), 1, self.audio_dim), device=self.device, dtype=self.dtype)
        
        # Log input shape and device for debugging
        if self.log_param_updates:
            logging.info(f"Audio input - shape: {audio.shape}, device: {audio.device}, dtype: {audio.dtype}")
        
        with torch.set_grad_enabled(not self.freeze_encoders):
            # Check if we're using FP16 and ensure consistent types
            if self.use_fp16 and audio.dtype != torch.float16:
                logging.info(f"Converting audio input from {audio.dtype} to float16")
                audio = audio.to(dtype=torch.float16)
            
            # If we're not using FP16 but the input is half precision, convert to float32
            if not self.use_fp16 and audio.dtype == torch.float16:
                logging.info(f"Converting audio input from float16 to float32")
                audio = audio.to(dtype=torch.float32)
            
            # If Whisper's parameters are not the same dtype as input, convert Whisper
            whisper_dtype = next(self.whisper.parameters()).dtype
            if audio.dtype != whisper_dtype:
                logging.warning(f"Type mismatch: audio is {audio.dtype} but Whisper is {whisper_dtype}. Converting Whisper.")
                # Move the whole Whisper model to the same dtype as the input
                self.whisper = self.whisper.to(dtype=audio.dtype)
                logging.info(f"Whisper model converted to {audio.dtype}")
            
            # Prepare attention mask if not provided
            if attention_mask is None:
                # Create mask where all elements are attended to (all 1s)
                attention_mask = torch.ones(audio.shape[0], audio.shape[1], dtype=torch.long, device=audio.device)
            
            # Ensure the attention mask has the correct device
            attention_mask = attention_mask.to(device=audio.device)
            
            # Log original audio size 
            batch_idx = getattr(self, '_current_batch_idx', 0)
            if batch_idx % 50 == 0:  # Changed from 10 to 50
                logging.info(f"[Batch {batch_idx}] AUDIO SEQ MONITOR: Raw input shape = {audio.shape}")
            
            # Get Whisper encoder output
            encoder_outputs = self.whisper.encoder(
                audio,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get the last hidden state
            last_hidden_state = encoder_outputs.last_hidden_state
            
            # Track audio sequence length statistics
            audio_seq_len = last_hidden_state.size(1)
            
            # Create or update audio sequence length stats
            if not hasattr(self, '_audio_seq_len_stats'):
                self._audio_seq_len_stats = {
                    'min': audio_seq_len,
                    'max': audio_seq_len,
                    'sum': audio_seq_len,
                    'count': 1,
                    'lengths': [audio_seq_len],
                }
            else:
                self._audio_seq_len_stats['min'] = min(self._audio_seq_len_stats['min'], audio_seq_len)
                self._audio_seq_len_stats['max'] = max(self._audio_seq_len_stats['max'], audio_seq_len)
                self._audio_seq_len_stats['sum'] += audio_seq_len
                self._audio_seq_len_stats['count'] += 1
                self._audio_seq_len_stats['lengths'].append(audio_seq_len)
                # Keep only the most recent 100 lengths to avoid memory growth
                if len(self._audio_seq_len_stats['lengths']) > 100:
                    self._audio_seq_len_stats['lengths'] = self._audio_seq_len_stats['lengths'][-100:]
            
            # Compute average sequence length
            avg_len = self._audio_seq_len_stats['sum'] / self._audio_seq_len_stats['count']
            
            # Log detailed audio sequence length info
            if batch_idx % 10 == 0:  # More frequent logging
                logging.info(f"[Batch {batch_idx}] AUDIO SEQ MONITOR: Whisper encoder output length = {audio_seq_len}")
                logging.info(f"[Batch {batch_idx}] AUDIO SEQ STATS: min={self._audio_seq_len_stats['min']}, "
                           f"max={self._audio_seq_len_stats['max']}, avg={avg_len:.1f}")
            
            # Project to LLM dimension
            features = self.audio_connector(last_hidden_state)
            
            # Log audio features after projection
            if batch_idx % 50 == 0:  # Changed from 10 to 50
                logging.info(f"[Batch {batch_idx}] AUDIO SEQ MONITOR: Projected features shape = {features.shape}")
            
            # Truncate sequence length to 512
            max_seq_len = 512
            if features.size(1) > max_seq_len:
                features = features[:, :max_seq_len, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :max_seq_len]
            
            return features, attention_mask
    
    def encode_video(self, video, attention_mask=None):
        """Encode video input using CLIP"""
        if self.clip is None:
            # For compatibility, return a dummy tensor with the right shape
            logging.warning("Video encoding attempted but CLIP model not loaded - returning dummy tensor")
            return torch.zeros((video.size(0), 1, self.video_dim), device=self.device, dtype=self.dtype)
        
        # Log input shape and device for debugging
        if self.log_param_updates:
            logging.info(f"Video input - shape: {video.shape}, device: {video.device}, dtype: {video.dtype}")
        
        # Process all frames together
        batch_size, seq_len = video.size(0), video.size(1)
        
        # Log original video size if in debug mode
        batch_idx = getattr(self, '_current_batch_idx', 0)
        if batch_idx % 50 == 0:
            logging.debug(f"[Batch {batch_idx}] Video encoder input: {batch_size} batches x {seq_len} frames")
        
        # Reshape to process all frames at once
        # CLIP expects input shape [batch_size, channels, height, width]
        # Our video is [batch_size, seq_len, channels, height, width]
        video_reshaped = video.view(batch_size * seq_len, video.size(2), video.size(3), video.size(4))
        
        with torch.set_grad_enabled(not self.freeze_encoders):
            # Get CLIP vision encoder output
            vision_outputs = self.clip(
                video_reshaped,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get the pooler output (or last hidden state)
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                pooled_output = vision_outputs.pooler_output
            else:
                # Use the [CLS] token embedding or the last hidden state if pooler not available
                pooled_output = vision_outputs.last_hidden_state[:, 0]
            
            # Reshape back to sequence form [batch_size, seq_len, feature_dim]
            features = pooled_output.view(batch_size, seq_len, -1)
            
            # Project to LLM dimension
            features = self.video_connector(features)
            
            # Log video features after projection if in debug mode
            if batch_idx % 50 == 0:
                logging.debug(f"[Batch {batch_idx}] Projected video features shape: {features.shape}")
            
            # We don't truncate here - preserve full sequence for fusion
            return features

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
        """Set up projections from encoder outputs to LLM input space"""
        from .modality_connector import create_modality_connector
        
        logging.info(f"Setting up projections for modality: {self.modality}")
        
        # Get connector type
        connector_type = self.connector_type
        logging.info(f"Using connector type: {connector_type}")
        
        # Create standard connectors (always create both for compatibility)
        # Audio projection (Whisper -> LLM)
        self.audio_connector = create_modality_connector(
            connector_type=self.connector_type,
            input_dim=self.audio_dim,
            output_dim=self.llm_dim,
            device=self.device,
            dtype=self.dtype,
            max_seq_len=self.max_seq_len
        )
        logging.info(f"Created audio projection ({self.connector_type}): {self.audio_dim} -> {self.llm_dim}")
        
        # Video projection (CLIP -> LLM)
        self.video_connector = create_modality_connector(
            connector_type=self.connector_type,
            input_dim=self.video_dim,
            output_dim=self.llm_dim,
            device=self.device,
            dtype=self.dtype,
            max_seq_len=self.max_seq_len
        )
        logging.info(f"Created video projection ({self.connector_type}): {self.video_dim} -> {self.llm_dim}")

    def _log_model_architecture(self):
        """Log the model architecture in a tabular format"""
        logging.info("\n" + "=" * 80)
        logging.info("CLIP-WHISPER MODEL ARCHITECTURE")
        logging.info("=" * 80)
        
        # Header
        logging.info(f"{'COMPONENT':<30} {'INPUT DIM':<15} {'OUTPUT DIM':<15} {'ACTIVE IN MODALITY':<20}")
        logging.info("-" * 80)
        
        # Get string representations for dimensions (handling None values)
        whisper_dim_str = str(self.audio_dim) if self.audio_dim is not None else "N/A"
        clip_dim_str = str(self.video_dim) if self.video_dim is not None else "N/A"
        llm_dim_str = str(self.llm_dim) if self.llm_dim is not None else "N/A"
        
        # LLM
        logging.info(f"{'LLM':<30} {'-':<15} {llm_dim_str:<15} {'all':<20}")
        
        # Encoders
        if self.modality in ["audio", "both"]:
            logging.info(f"{'Audio Encoder (Whisper)':<30} {'-':<15} {whisper_dim_str:<15} {'audio, both':<20}")
        
        if self.modality in ["video", "both"]:
            logging.info(f"{'Video Encoder (CLIP)':<30} {'-':<15} {clip_dim_str:<15} {'video, both':<20}")
        
        # Connectors
        if self.modality in ["audio", "both"] and hasattr(self, 'audio_connector') and self.audio_connector is not None:
            logging.info(f"{'Audio Connector':<30} {whisper_dim_str:<15} {llm_dim_str:<15} {'audio, both':<20}")
        
        if self.modality in ["video", "both"] and hasattr(self, 'video_connector') and self.video_connector is not None:
            logging.info(f"{'Video Connector':<30} {clip_dim_str:<15} {llm_dim_str:<15} {'video, both':<20}")
        
        logging.info("=" * 80)
        
        # Display active components summary
        logging.info("\nACTIVE COMPONENTS FOR MODALITY: {}".format(self.modality.upper()))
        if self.modality in ['audio', 'both']:
            logging.info("- Using WHISPER for audio encoding")
        if self.modality in ['video', 'both']:
            logging.info("- Using CLIP for video encoding")
        if self.modality == 'both':
            logging.info("- Using connector for multimodal fusion")
            
        # Always using a connector to project to LLM dimension
        logging.info(f"- All features projected to LLM dimension: {llm_dim_str}")
        
        logging.info("=" * 80 + "\n")

    def generate(
        self,
        audio=None,
        video=None,
        prompt=None,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        **kwargs
    ):
        """Generate text from audio/video embeddings"""
        try:
            logging.info("Starting generation process")
            
            # Create inputs_embeds and attention_mask using our encoders
            if self.modality == "audio":
                if audio is None:
                    raise ValueError("Audio modality selected but no audio input provided")
                encoder_output = self.encode_audio(audio)
            elif self.modality == "video":
                if video is None:
                    raise ValueError("Video modality selected but no video input provided")
                encoder_output = self.encode_video(video)
            else:  # both modalities
                if audio is None or video is None:
                    raise ValueError("Both modalities selected but inputs are missing")
                
                audio_features = self.encode_audio(audio)
                video_features = self.encode_video(video)
                
                # Ensure both features have the same sequence length
                max_len = min(
                    self.max_seq_len,
                    max(audio_features.size(1), video_features.size(1))
                )
                audio_features = self._pad_or_truncate(audio_features, max_len)
                video_features = self._pad_or_truncate(video_features, max_len)
                
                # Weighted sum of features
                encoder_output = (
                    self.fusion_scale * audio_features +
                    (1 - self.fusion_scale) * video_features
                )
            
            # Track batch dimension for inputs to LLM
            batch_size = encoder_output.size(0)
            
            # Create attention mask (all 1s for the encoder output)
            attention_mask = torch.ones(
                (batch_size, encoder_output.size(1)),
                dtype=torch.long,
                device=self.device
            )
            
            # Handle prompt for LLM input
            if prompt is not None:
                # Process the provided prompt
                if isinstance(prompt, str):
                    # Convert text prompt to token IDs
                    prompt_ids = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_seq_len
                    ).input_ids.to(self.device)
                else:
                    # Assume it's already tokenized
                    prompt_ids = prompt.to(self.device)
                
                # Convert tokens to embeddings
                embedding_layer = self.llm.get_input_embeddings()
                prompt_embeds = embedding_layer(prompt_ids)
                
                # Ensure batch dimension matches
                if prompt_embeds.size(0) == 1 and batch_size > 1:
                    prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
                
                # Concatenate prompt and encoder output
                inputs_embeds = torch.cat([prompt_embeds, encoder_output], dim=1)
                
                # Update attention mask
                prompt_attention = torch.ones(
                    (batch_size, prompt_embeds.size(1)),
                    dtype=torch.long,
                    device=self.device
                )
                attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
                
                logging.info(f"Using provided prompt. Input shape: {inputs_embeds.shape}")
            else:
                # Add a default transcription prompt for consistent behavior
                default_prompt = "Transcribe the speech from this audio and video input: "
                
                if hasattr(self, 'tokenizer'):
                    # Ensure pad token is set
                    if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Convert prompt to token IDs
                    prompt_ids = self.tokenizer(
                        default_prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_seq_len
                    ).input_ids.to(self.device)
                    
                    # Get embeddings
                    embedding_layer = self.llm.get_input_embeddings()
                    prompt_embeds = embedding_layer(prompt_ids)
                    
                    # Ensure batch dimension matches
                    if prompt_embeds.size(0) == 1 and batch_size > 1:
                        prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
                    
                    # Concatenate prompt and encoder output
                    inputs_embeds = torch.cat([prompt_embeds, encoder_output], dim=1)
                    
                    # Update attention mask
                    prompt_attention = torch.ones(
                        (batch_size, prompt_embeds.size(1)),
                        dtype=torch.long,
                        device=self.device
                    )
                    attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
                    
                    logging.info(f"Using default prompt. Input shape: {inputs_embeds.shape}")
                else:
                    # No tokenizer available, use encoder output directly
                    inputs_embeds = encoder_output
                    logging.warning("No tokenizer available, cannot add default prompt")
            
            # Log the final input shape before generation
            if 'inputs_embeds' in locals():
                logging.info(f"Generating with inputs shape: {inputs_embeds.shape}")
            else:
                inputs_embeds = encoder_output
                logging.info(f"Generating with inputs shape: {encoder_output.shape}")
            
            # Generate text using the LLM
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                **kwargs
            )
            
            return outputs
            
        except Exception as e:
            logging.error(f"Error in generate method: {e}")
            logging.error(traceback.format_exc())
            raise 