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
            max_seq_len: Maximum sequence length for encoder output
            fusion_scale: Weight for audio in fusion (0.5 = equal weight)
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
        
        # Initialize all sub-models
        self.llm, self.tokenizer = self._track_component_memory("LLM",
                    lambda: self._load_llm(llm_path, use_lora, lora_r, lora_alpha, lora_dropout, freeze_llm, use_4bit))
        self.whisper, self.whisper_processor = self._track_component_memory("Whisper",
                            lambda: self._load_whisper_model(whisper_model, freeze_encoders))
        self.clip, self.clip_processor = self._track_component_memory("CLIP",
                         lambda: self._load_clip_model(clip_model, freeze_encoders))
        
        # Get dimensions of encoders and LLM
        self.audio_dim = self.whisper.config.d_model  # Whisper dimension
        self.video_dim = self.clip.config.hidden_size  # CLIP dimension
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
        }
        
        # Count and log trainable parameters by component
        self._log_parameter_count()

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
        """Forward pass"""
        # Save batch index for logging if provided
        if 'batch_idx' in kwargs:
            self._current_batch_idx = kwargs['batch_idx']
        
        batch_idx = getattr(self, '_current_batch_idx', 0)
            
        try:
            # Start memory tracking
            fw_memory_start = self._get_gpu_memory_usage()
            
            # Log the modality that's being used 
            if batch_idx % 25 == 0:
                logging.info(f"[Batch {batch_idx}] Forward pass using modality: {self.modality}")
            
            batch_size = audio.size(0) if audio is not None else video.size(0)
            
            logging.debug(f"Forward pass with batch size {batch_size}")
            
            # Input validation
            if audio is None and video is None:
                raise ValueError("Both audio and video inputs cannot be None")
                
            if audio is not None:
                logging.debug(f"Audio input shape: {audio.shape}")
            if video is not None:
                logging.debug(f"Video input shape: {video.shape}")
            
            # Encode audio and/or video
            audio_features = None
            video_features = None
            
            # Track memory before encoding
            encoder_memory_start = self._get_gpu_memory_usage()
            
            # Audio encoding
            if (self.modality == "audio" or self.modality == "both") and audio is not None:
                audio_features = self.encode_audio(audio)
                # Log initial audio features size before any truncation in fusion
                if batch_idx % 25 == 0:
                    logging.info(f"[Batch {batch_idx}] Pre-fusion audio features shape: {audio_features.shape}")
            
            # Video encoding
            if (self.modality == "video" or self.modality == "both") and video is not None:
                video_features = self.encode_video(video)
                # Log initial video features size before any truncation in fusion
                if batch_idx % 25 == 0:
                    logging.info(f"[Batch {batch_idx}] Pre-fusion video features shape: {video_features.shape}")
            
            # Track memory after encoding
            encoder_memory_end = self._get_gpu_memory_usage()
            encoder_memory_used = encoder_memory_end["allocated"] - encoder_memory_start["allocated"]
            if batch_idx % 50 == 0:
                logging.info(f"[Batch {batch_idx}] Encoder memory usage: {encoder_memory_used:.2f} MB")
                
            # Combine features based on modality
            if self.modality == "audio":
                encoder_out = audio_features
                # Log audio-only mode sequence info
                if batch_idx % 10 == 0:
                    logging.info(f"[Batch {batch_idx}] AUDIO-ONLY MODE: Using audio features with shape {audio_features.shape}")
            elif self.modality == "video":
                encoder_out = video_features
                # Log video-only mode sequence info
                if batch_idx % 10 == 0:
                    logging.info(f"[Batch {batch_idx}] VIDEO-ONLY MODE: Using video features with shape {video_features.shape}")
            else:  # both modalities
                # Log original sequence lengths before fusion
                if batch_idx % 10 == 0 and audio_features is not None and video_features is not None:
                    logging.info(f"[Batch {batch_idx}] FUSION SEQ MONITOR: Pre-fusion audio length = {audio_features.size(1)}, video length = {video_features.size(1)}")
                    logging.info(f"[Batch {batch_idx}] FUSION SEQ MONITOR: max_seq_len = {self.max_seq_len}")
                
                # Ensure both features have the same sequence length,
                # but also respect the maximum sequence length limit
                max_len = min(
                    self.max_seq_len,
                    max(
                        audio_features.size(1) if audio_features is not None else 0,
                        video_features.size(1) if video_features is not None else 0
                    )
                )
                
                # Log the selected fusion length
                if batch_idx % 10 == 0:
                    if max_len < self.max_seq_len:
                        logging.info(f"[Batch {batch_idx}] FUSION SEQ MONITOR: Using natural max length {max_len} (< max_seq_len {self.max_seq_len})")
                    else:
                        logging.info(f"[Batch {batch_idx}] FUSION SEQ MONITOR: Truncating to max_seq_len {max_len}")
                
                # Log if we're truncating during fusion
                if audio_features is not None and audio_features.size(1) > max_len:
                    # Get the current batch index if available for logging
                    batch_idx = getattr(self, '_current_batch_idx', 0)
                    # Log less frequently to reduce clutter (only every 25 batches)
                    if batch_idx % 10 == 0:
                        pct_kept = (max_len / audio_features.size(1)) * 100
                        logging.info(f"[Batch {batch_idx}] FUSION SEQ MONITOR: Audio truncated from {audio_features.size(1)} → {max_len} ({pct_kept:.1f}% kept)")
                
                if video_features is not None and video_features.size(1) > max_len:
                    # Get the current batch index if available for logging
                    batch_idx = getattr(self, '_current_batch_idx', 0)
                    # Log less frequently to reduce clutter (only every 25 batches)
                    if batch_idx % 10 == 0:
                        pct_kept = (max_len / video_features.size(1)) * 100
                        logging.info(f"[Batch {batch_idx}] FUSION SEQ MONITOR: Video truncated from {video_features.size(1)} → {max_len} ({pct_kept:.1f}% kept)")
                
                # Pad or truncate to match
                audio_features = self._pad_or_truncate(audio_features, max_len)
                video_features = self._pad_or_truncate(video_features, max_len)
                
                # Double-check that fusion result respects max_seq_len
                if audio_features.size(1) != max_len or video_features.size(1) != max_len:
                    logging.warning(f"Unexpected feature length after padding/truncation - Audio: {audio_features.size(1)}, Video: {video_features.size(1)}, Expected: {max_len}")
                    audio_features = self._pad_or_truncate(audio_features, max_len)
                    video_features = self._pad_or_truncate(video_features, max_len)
                
                # Weighted sum of features
                encoder_out = (
                    self.fusion_scale * audio_features +
                    (1 - self.fusion_scale) * video_features
                )
                
                # Log fusion result
                if batch_idx % 10 == 0:
                    logging.info(f"[Batch {batch_idx}] FUSION SEQ MONITOR: Final fused features shape = {encoder_out.shape}")
                
                # Double-check that fusion result respects max_seq_len
                if encoder_out.size(1) != max_len:
                    logging.warning(f"Unexpected encoder output length after fusion: {encoder_out.size(1)} (expected {max_len})")
                    encoder_out = self._pad_or_truncate(encoder_out, max_len)
            
            # Log final encoder output size before any LLM processing
            if batch_idx % 10 == 0:
                logging.info(f"[Batch {batch_idx}] FINAL SEQ MONITOR: Encoder output before LLM = {encoder_out.shape}")
            
            # Track sequence length stats across all batches - only use the length value after all adjustments
            self._track_sequence_length(encoder_out.size(1))
            
            # Ensure encoder output matches LLM's expected dtype
            llm_input_dtype = next(self.llm.parameters()).dtype
            if encoder_out.dtype != llm_input_dtype:
                logging.info(f"Converting encoder output from {encoder_out.dtype} to {llm_input_dtype}")
                encoder_out = encoder_out.to(llm_input_dtype)
            
            # Create initial attention mask (all 1s)
            attention_mask = torch.ones((batch_size, encoder_out.size(1)), dtype=torch.long, device=self.device)
                
            # Track memory for LLM forward pass at the appropriate place
            llm_memory_start = self._get_gpu_memory_usage()
            
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
                    # Get the current global step from kwargs for logging
                    batch_idx = kwargs.get('batch_idx', 0)
                    
                    # Log occasionally (every 25 batches) to reduce clutter but maintain visibility
                    if batch_idx % 25 == 0:
                        logging.info(f"[Batch {batch_idx}] Label sequence length ({labels.size(1)}) doesn't match encoder output ({encoder_out.size(1)})")
                    else:
                        logging.debug(f"Label sequence length ({labels.size(1)}) doesn't match encoder output ({encoder_out.size(1)})")
                    
                    # Must resize labels to match encoder output exactly
                    if labels.size(1) < encoder_out.size(1):
                        # Pad labels with -100 (ignored in loss calculation)
                        padding = torch.full(
                            (batch_size, encoder_out.size(1) - labels.size(1)), 
                            -100,  # Padding token ID that is ignored in loss calculation
                            dtype=labels.dtype, 
                            device=labels.device
                        )
                        labels = torch.cat([labels, padding], dim=1)
                    else:
                        # Truncate labels to match encoder output length
                        labels = labels[:, :encoder_out.size(1)]
                    
                    # Log resizing at debug level
                    logging.debug(f"Resized labels to match encoder output: {labels.shape}")
                
                # Check if batch sizes match
                if labels.size(0) != encoder_out.size(0):
                    raise ValueError(f"Label batch size ({labels.size(0)}) doesn't match encoder output ({encoder_out.size(0)})")
                
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
            
            # Track memory after LLM forward pass
            llm_memory_end = self._get_gpu_memory_usage()
            llm_memory_used = llm_memory_end["allocated"] - llm_memory_start["allocated"]
            
            # Calculate total memory usage for the forward pass
            fw_memory_end = self._get_gpu_memory_usage()
            fw_memory_used = fw_memory_end["allocated"] - fw_memory_start["allocated"]
            
            # Log memory usage less frequently
            if batch_idx % 50 == 0:
                logging.info(f"[Batch {batch_idx}] GPU Memory - LLM forward: {llm_memory_used:.2f} MB, Encoder: {encoder_memory_used:.2f} MB, Total: {fw_memory_used:.2f} MB")
            
            return outputs
        
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            logging.error(traceback.format_exc())
            raise
    
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
                "device": self.device,
                "dtype": self.dtype,
            }, os.path.join(output_dir, "model_config.json"))
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
    def from_pretrained(cls, model_dir):
        """Load model from saved checkpoint"""
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist")
        
        # Load model configuration
        config_path = os.path.join(model_dir, "model_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Model config {config_path} does not exist")
        
        with open(config_path, "rb") as f:
            config = torch.load(f)
        
        # Create model with the saved configuration
        model = cls(
            llm_path=os.path.join(model_dir, "llm"),
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
        try:
            processor = WhisperProcessor.from_pretrained(whisper_model)
            model = WhisperModel.from_pretrained(whisper_model)
            if freeze_encoders:
                self._freeze_model(model)
            return model, processor
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            raise

    def _load_clip_model(self, clip_model, freeze_encoders):
        """Load the CLIP model for visual encoding"""
        logging.info(f"Loading CLIP model: {clip_model}")
        try:
            processor = CLIPProcessor.from_pretrained(clip_model)
            model = CLIPVisionModel.from_pretrained(clip_model)
            if freeze_encoders:
                self._freeze_model(model)
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
        
        # Configure 4-bit quantization if requested
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
                )
                
                logging.info("Successfully loaded LLM with 4-bit quantization")
            except ImportError:
                logging.warning("BitsAndBytes not available for 4-bit quantization, falling back to standard loading")
                llm = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto")
            except Exception as e:
                logging.error(f"Error loading LLM with 4-bit quantization: {e}")
                logging.warning("Falling back to standard loading")
                llm = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto")
        else:
            # Standard loading without quantization
            llm = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        
        # Apply LoRA if requested
        if use_lora:
            logging.info(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            
            # Configure target modules based on model type
            if 'llama' in llm_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                # Default target modules for other model types
                target_modules = ["query", "key", "value", "dense"]
                
            # Create LoRA config
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            
            # Apply LoRA to model
            llm = get_peft_model(llm, peft_config)
            logging.info("LoRA applied successfully")
            
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
        # Count parameters by component
        whisper_params = sum(p.numel() for p in self.whisper.parameters())
        clip_params = sum(p.numel() for p in self.clip.parameters())
        llm_params = sum(p.numel() for p in self.llm.parameters())
        audio_connector_params = sum(p.numel() for p in self.audio_connector.parameters())
        video_connector_params = sum(p.numel() for p in self.video_connector.parameters())
        
        # Count trainable parameters by component
        whisper_trainable = sum(p.numel() for p in self.whisper.parameters() if p.requires_grad)
        clip_trainable = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        llm_trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        audio_connector_trainable = sum(p.numel() for p in self.audio_connector.parameters() if p.requires_grad)
        video_connector_trainable = sum(p.numel() for p in self.video_connector.parameters() if p.requires_grad)
        
        # Total counts
        total_params = whisper_params + clip_params + llm_params + \
                       audio_connector_params + video_connector_params
                        
        total_trainable = whisper_trainable + clip_trainable + llm_trainable + \
                          audio_connector_trainable + video_connector_trainable
        
        # Log the counts in a nice table format
        logging.info("=" * 80)
        logging.info("Parameter counts by component:")
        logging.info("-" * 80)
        logging.info(f"{'Component':<20} {'Total':<15} {'Trainable':<15} {'% Trainable':<10}")
        logging.info("-" * 80)
        logging.info(f"{'Whisper':<20} {whisper_params:,d} {whisper_trainable:,d} {100*whisper_trainable/max(1, whisper_params):.2f}%")
        logging.info(f"{'CLIP':<20} {clip_params:,d} {clip_trainable:,d} {100*clip_trainable/max(1, clip_params):.2f}%")
        logging.info(f"{'LLM':<20} {llm_params:,d} {llm_trainable:,d} {100*llm_trainable/max(1, llm_params):.2f}%")
        logging.info(f"{'Audio Connector':<20} {audio_connector_params:,d} {audio_connector_trainable:,d} {100*audio_connector_trainable/max(1, audio_connector_params):.2f}%")
        logging.info(f"{'Video Connector':<20} {video_connector_params:,d} {video_connector_trainable:,d} {100*video_connector_trainable/max(1, video_connector_params):.2f}%")
        logging.info("-" * 80)
        logging.info(f"{'TOTAL':<20} {total_params:,d} {total_trainable:,d} {100*total_trainable/max(1, total_params):.2f}%")
        logging.info("=" * 80)
        
        # Log active components based on modality
        logging.info(f"Active components for modality '{self.modality}':")
        if self.modality in ["audio", "both"]:
            logging.info(f"- Whisper: {whisper_params:,d} parameters {'(frozen)' if not whisper_trainable else ''}")
            logging.info(f"- Audio Connector: {audio_connector_params:,d} parameters {'(frozen)' if not audio_connector_trainable else ''}")
        if self.modality in ["video", "both"]:
            logging.info(f"- CLIP: {clip_params:,d} parameters {'(frozen)' if not clip_trainable else ''}")
            logging.info(f"- Video Connector: {video_connector_params:,d} parameters {'(frozen)' if not video_connector_trainable else ''}")
        logging.info(f"- LLM: {llm_params:,d} parameters {'(frozen)' if not llm_trainable else ''}")
    
    def encode_audio(self, audio, attention_mask=None):
        """Encode audio using Whisper"""
        if self.modality == "video":
            logging.warning("Trying to encode audio in video-only mode, returning dummy tensor")
            return torch.zeros((audio.size(0), 1, self.llm_dim), device=self.device, dtype=self.dtype)
        
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
            if batch_idx % 10 == 0:  # More frequent logging (every 10 batches)
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
                           f"max={self._audio_seq_len_stats['max']}, avg={avg_len:.1f}, "
                           f"recent={self._audio_seq_len_stats['lengths'][-5:]}")
            
            # Project to LLM dimension
            features = self.audio_connector(last_hidden_state)
            
            # Log audio features after projection
            if batch_idx % 10 == 0:
                logging.info(f"[Batch {batch_idx}] AUDIO SEQ MONITOR: Projected features shape = {features.shape}")
            
            # We don't truncate here - preserve full sequence for fusion
            return features
    
    def encode_video(self, video, attention_mask=None):
        """Encode video using CLIP"""
        if self.modality == "audio":
            logging.warning("Trying to encode video in audio-only mode, returning dummy tensor")
            return torch.zeros((video.size(0), 1, self.llm_dim), device=self.device, dtype=self.dtype)
        
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
        """Create projections from encoder outputs to LLM input dimension"""
        # Audio projection (Whisper -> LLM)
        self.audio_connector = ModalityConnector(
            input_dim=self.audio_dim,
            output_dim=self.llm_dim,
            device=self.device,
            dtype=self.dtype
        )
        logging.info(f"Created audio projection: {self.audio_dim} -> {self.llm_dim}")
        
        # Video projection (CLIP -> LLM)
        self.video_connector = ModalityConnector(
            input_dim=self.video_dim,
            output_dim=self.llm_dim,
            device=self.device,
            dtype=self.dtype
        )
        logging.info(f"Created video projection: {self.video_dim} -> {self.llm_dim}")
        
    def _log_model_architecture(self):
        """Log model architecture details"""
        # Print detailed dimension table for all components
        dim_banner = "-" * 80
        logging.info(f"\n{dim_banner}")
        logging.info(f"{'COMPONENT':<30} {'INPUT DIM':<15} {'OUTPUT DIM':<15} {'ACTIVE IN MODALITY':<20}")
        logging.info(f"{dim_banner}")
        
        # Audio components
        logging.info(f"{'Audio Encoder (Whisper)':<30} {'-':<15} {self.audio_dim:<15} {'audio, both':<20}")
        logging.info(f"{'Audio Connector':<30} {self.audio_dim:<15} {self.llm_dim:<15} {'audio, both':<20}")
        
        # Video components
        logging.info(f"{'Video Encoder (CLIP)':<30} {'-':<15} {self.video_dim:<15} {'video, both':<20}")
        logging.info(f"{'Video Connector':<30} {self.video_dim:<15} {self.llm_dim:<15} {'video, both':<20}")
        
        # LLM
        logging.info(f"{'LLM':<30} {self.llm_dim:<15} {'-':<15} {'all':<20}")
        logging.info(f"{dim_banner}\n")

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
                
                # Log if we're truncating during fusion
                if audio_features.size(1) > max_len:
                    logging.debug(f"Truncating audio features from {audio_features.size(1)} to {max_len} during fusion")
                
                if video_features.size(1) > max_len:
                    logging.debug(f"Truncating video features from {video_features.size(1)} to {max_len} during fusion")
                
                audio_features = self._pad_or_truncate(audio_features, max_len)
                video_features = self._pad_or_truncate(video_features, max_len)
                
                # Apply fusion
                encoder_output = (
                    self.fusion_scale * audio_features +
                    (1 - self.fusion_scale) * video_features
                )
                
                # Double-check that fusion result respects max_seq_len
                if encoder_output.size(1) != max_len:
                    logging.warning(f"Unexpected encoder output length after fusion: {encoder_output.size(1)} (expected {max_len})")
                    encoder_output = self._pad_or_truncate(encoder_output, max_len)
            
            # Apply sequence length limit
            if encoder_output.size(1) > self.max_seq_len:
                # Get the current batch index if available for logging
                batch_idx = getattr(self, '_current_batch_idx', 0)
                # Log less frequently to reduce clutter (only every 25 batches)
                if batch_idx % 25 == 0:
                    logging.info(f"[Batch {batch_idx}] Encoder output truncated from {encoder_output.size(1)} to {self.max_seq_len}")
                encoder_output = encoder_output[:, :self.max_seq_len, :]
                logging.debug(f"Truncated encoder output to {self.max_seq_len}")
            
            # Explicitly verify the sequence length after truncation
            if encoder_output.size(1) != self.max_seq_len:
                # Get the current batch index if available for logging
                batch_idx = getattr(self, '_current_batch_idx', 0)
                if batch_idx % 25 == 0:
                    logging.info(f"[Batch {batch_idx}] After truncation, encoder output length ({encoder_output.size(1)}) doesn't match max_seq_len ({self.max_seq_len})")
                # Force the correct sequence length if it still doesn't match
                if encoder_output.size(1) > self.max_seq_len:
                    encoder_output = encoder_output[:, :self.max_seq_len, :]
                    logging.debug(f"Forcibly truncated encoder output to max_seq_len {self.max_seq_len}")
            
            # Get batch size and sequence length
            batch_size, seq_len = encoder_output.size(0), encoder_output.size(1)
            
            # Create attention mask
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)
            
            # Handle prompt if provided
            if prompt is not None:
                embedding_layer = self.llm.get_input_embeddings()
                prompt_embeds = embedding_layer(prompt.to(self.device))
                
                # Ensure batch dimensions match
                if prompt_embeds.size(0) == 1 and encoder_output.size(0) > 1:
                    prompt_embeds = prompt_embeds.expand(encoder_output.size(0), -1, -1)
                
                # Concatenate along sequence dimension
                inputs_embeds = torch.cat([prompt_embeds, encoder_output], dim=1)
                
                # Update attention mask to include prompt tokens
                attention_mask = torch.ones(
                    (batch_size, inputs_embeds.size(1)), 
                    dtype=torch.long, 
                    device=self.device
                )
            else:
                inputs_embeds = encoder_output
            
            # Ensure input embeddings match LLM's expected dtype
            llm_dtype = next(self.llm.parameters()).dtype
            if inputs_embeds.dtype != llm_dtype:
                inputs_embeds = inputs_embeds.to(llm_dtype)
            
            # Set up generation parameters
            generation_config = {
                "max_length": max_length,
                "temperature": temperature,
                "do_sample": do_sample,
                **kwargs
            }
            
            logging.info(f"Generating with inputs shape: {inputs_embeds.shape}")
            
            # Generate text
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_config
            )
            
            return outputs
            
        except Exception as e:
            logging.error(f"Error in generate method: {e}")
            logging.error(traceback.format_exc())
            raise 