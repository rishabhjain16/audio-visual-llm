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
    AutoModelForCausalLM
)
import datetime

class ModalityConnector(nn.Module):
    """Linear projection for modality encoding to LLM dimension"""

    def __init__(self, input_dim, output_dim, device="cuda", dtype=torch.float32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        
        logging.info(f"Creating ModalityConnector: input_dim={input_dim}, output_dim={output_dim}, dtype={dtype}")
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Use standard PyTorch initialization 
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.bias)
        
        # Move to specified device and dtype
        self.linear = self.linear.to(device=device, dtype=dtype)
        
        logging.info(f"Created ModalityConnector with standard initialization")

    def forward(self, x):
        # Ensure input has correct dtype
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return self.linear(x)


class SimpleAVSRModel(nn.Module):
    """
    A simple Audio-Visual Speech Recognition model inspired by SpeechLLM.
    Uses Whisper for audio, CLIP for video, and Llama for language modeling.
    """
    
    def __init__(
        self,
        llm_path: str = "meta-llama/Llama-2-7b-chat-hf",
        whisper_model: str = "openai/whisper-medium",
        clip_model: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = False,
        use_lora: bool = True,
        use_4bit: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_encoders: bool = True,
        freeze_llm: bool = False,
        modality: str = "both",
        max_seq_len: int = 256,
        fusion_scale: float = 0.5,
    ):
        """Initialize the AVSR model"""
        super().__init__()
        
        # Set modality with clear logging
        self.modality = modality
        modality_banner = "=" * 80
        logging.info(f"\n{modality_banner}")
        if modality == "audio":
            logging.info(f"USING AUDIO-ONLY MODALITY")
            logging.info(f"Only audio encoder and audio connector will be used")
        elif modality == "video":
            logging.info(f"USING VIDEO-ONLY MODALITY")
            logging.info(f"Only video encoder and video connector will be used")
        elif modality == "both":
            logging.info(f"USING COMBINED AUDIO+VIDEO MODALITY")
            logging.info(f"Both encoders will be used with fusion_scale={fusion_scale}")
        else:
            logging.warning(f"Unknown modality: {modality}, defaulting to 'both'")
            self.modality = "both"
        logging.info(f"{modality_banner}\n")
        
        # Store configuration options
        self.device = device
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.freeze_encoders = freeze_encoders
        self.freeze_llm = freeze_llm
        self.max_seq_len = max_seq_len
        self.fusion_scale = fusion_scale
        
        # Store initialization parameters
        self.use_fp16 = use_fp16
        self.use_lora = use_lora
        self.use_4bit = use_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Set dtype based on use_fp16
        logging.info(f"Using dtype: {self.dtype}")
        
        # Load models and processors
        logging.info("Loading models and processors...")
        self.whisper, self.whisper_processor = self._load_whisper_model(whisper_model)
        self.clip, self.clip_processor = self._load_clip_model(clip_model)
        
        # Get model dimensions
        self.audio_dim = self.whisper.config.d_model  # Usually 1024 for whisper-medium
        self.video_dim = self.clip.config.hidden_size  # Usually 768 for CLIP
        logging.info(f"Model dimensions - Audio: {self.audio_dim}, Video: {self.video_dim}")
        
        # Load LLM and tokenizer
        logging.info(f"Loading LLM from {llm_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = self._load_llm(llm_path)
        
        # Detect LLM dimension - handle both Llama-2 (4096) and Llama-3 (2048/3072/4096)
        self.llm_dim = self.llm.config.hidden_size
        logging.info(f"LLM dimension detected: {self.llm_dim}")
        
        # Create connectors for each modality (directly to LLM dimension)
        # Using consistent dtype (float32) for stability
        connector_dtype = torch.float32
        
        # Print detailed dimension table for all components
        dim_banner = "-" * 80
        logging.info(f"\n{dim_banner}")
        logging.info(f"{'COMPONENT':<30} {'INPUT DIM':<15} {'OUTPUT DIM':<15} {'DTYPE':<10} {'ACTIVE IN MODALITY':<20}")
        logging.info(f"{dim_banner}")
        
        # Audio components
        self.audio_connector = ModalityConnector(
            input_dim=self.audio_dim,
            output_dim=self.llm_dim,
            device=self.device,
            dtype=connector_dtype
        )
        logging.info(f"{'Audio Encoder (Whisper)':<30} {'-':<15} {self.audio_dim:<15} {next(self.whisper.parameters()).dtype} {'audio, both':<20}")
        logging.info(f"{'Audio Connector':<30} {self.audio_dim:<15} {self.llm_dim:<15} {connector_dtype} {'audio, both':<20}")
        
        # Video components
        self.video_connector = ModalityConnector(
            input_dim=self.video_dim,
            output_dim=self.llm_dim,
            device=self.device,
            dtype=connector_dtype
        )
        logging.info(f"{'Video Encoder (CLIP)':<30} {'-':<15} {self.video_dim:<15} {next(self.clip.parameters()).dtype} {'video, both':<20}")
        logging.info(f"{'Video Connector':<30} {self.video_dim:<15} {self.llm_dim:<15} {connector_dtype} {'video, both':<20}")
        
        # LLM
        logging.info(f"{'LLM':<30} {self.llm_dim:<15} {'-':<15} {next(self.llm.parameters()).dtype} {'all':<20}")
        logging.info(f"{dim_banner}\n")
        
        # Freeze encoders if specified
        if self.freeze_encoders:
            self._freeze_model(self.whisper)
            self._freeze_model(self.clip)
            logging.info("Encoders frozen")
        
        # Move entire model to device
        self.to(device)
        
        # Log model configuration
        logging.info(f"Model initialized with:"
                    f"\n  Modality: {modality}"
                    f"\n  Audio dim: {self.audio_dim}"
                    f"\n  Video dim: {self.video_dim}"
                    f"\n  LLM dim: {self.llm_dim}"
                    f"\n  Device: {device}"
                    f"\n  FP16: {use_fp16}"
                    f"\n  LoRA: {use_lora}"
                    f"\n  4-bit: {use_4bit}")
        
        # Save initialization parameters BEFORE loading LLM
        self.config = {
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

    def _pad_or_truncate(self, x, target_len):
        """Pad or truncate sequence to target length"""
        curr_len = x.size(1)
        if curr_len > target_len:
            return x[:, :target_len, :]
        elif curr_len < target_len:
            padding = torch.zeros(
                (x.size(0), target_len - curr_len, x.size(2)),
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
            # Log the modality that's being used with a clear banner for easy identification
            modality_banner = "#" * 50
            logging.info(f"\n{modality_banner}")
            logging.info(f"# FORWARD PASS USING MODALITY: {self.modality.upper()} #")
            
            # Verify expected inputs based on modality for debugging
            if self.modality == "audio" and audio is None:
                logging.warning("Audio modality selected but no audio input provided")
            elif self.modality == "video" and video is None:
                logging.warning("Video modality selected but no video input provided")
            elif self.modality == "both" and (audio is None or video is None):
                if audio is None:
                    logging.warning("Both modality selected but no audio input provided")
                if video is None:
                    logging.warning("Both modality selected but no video input provided")
            
            logging.info(f"{modality_banner}\n")
            
            # Log input shapes for debugging
            if audio is not None:
                logging.info(f"Input audio shape: {audio.shape}, dtype: {audio.dtype}")
            if video is not None:
                logging.info(f"Input video shape: {video.shape}, dtype: {video.dtype}")
            if labels is not None:
                logging.info(f"Input labels shape: {labels.shape}")
            
            # Encode audio if provided
            audio_features = None
            if audio is not None and self.modality in ["audio", "both"]:
                logging.info(f"Processing audio input for modality: {self.modality}")
                # Handle NaN/Inf values
                if torch.isnan(audio).any() or torch.isinf(audio).any():
                    audio = torch.nan_to_num(audio)
                
                audio_features = self.encode_audio(audio)
                
                # Handle NaN/Inf values
                if audio_features is not None and (torch.isnan(audio_features).any() or torch.isinf(audio_features).any()):
                    audio_features = torch.nan_to_num(audio_features)
            elif self.modality == "audio" and audio is None:
                logging.warning("Audio modality selected but no audio input provided")
            
            # Encode video if provided
            video_features = None
            if video is not None and self.modality in ["video", "both"]:
                logging.info(f"Processing video input for modality: {self.modality}")
                # Handle NaN/Inf values
                if torch.isnan(video).any() or torch.isinf(video).any():
                    video = torch.nan_to_num(video)
                
                video_features = self.encode_video(video)
                
                # Handle NaN/Inf values
                if video_features is not None and (torch.isnan(video_features).any() or torch.isinf(video_features).any()):
                    video_features = torch.nan_to_num(video_features)
            elif self.modality == "video" and video is None:
                logging.warning("Video modality selected but no video input provided")
            
            # Determine which modality to use and project to LLM dimension
            encoder_out = None
            batch_size = None
            seq_len = None
            
            # Audio only path
            if audio_features is not None and (self.modality == "audio" or (self.modality == "both" and video_features is None)):
                # Truncate to max_seq_len
                audio_features = audio_features[:, :self.max_seq_len, :]
                logging.info(f"Audio features before connector: shape={audio_features.shape}, dtype={audio_features.dtype}")
                encoder_out = self.audio_connector(audio_features)
                logging.info(f"Audio features after connector: shape={encoder_out.shape}, dtype={encoder_out.dtype}")
                batch_size, seq_len = encoder_out.size(0), encoder_out.size(1)
                path_banner = "=" * 80
                logging.info(f"\n{path_banner}")
                logging.info(f"USING AUDIO-ONLY PATH (modality={self.modality})")
                logging.info(f"Encoder output shape: {encoder_out.shape}, dtype: {encoder_out.dtype}")
                logging.info(f"{path_banner}")
            
            # Video only path
            elif video_features is not None and (self.modality == "video" or (self.modality == "both" and audio_features is None)):
                # Truncate to max_seq_len
                video_features = video_features[:, :self.max_seq_len, :]
                logging.info(f"Video features before connector: shape={video_features.shape}, dtype={video_features.dtype}")
                encoder_out = self.video_connector(video_features)
                logging.info(f"Video features after connector: shape={encoder_out.shape}, dtype={encoder_out.dtype}")
                batch_size, seq_len = encoder_out.size(0), encoder_out.size(1)
                path_banner = "=" * 80
                logging.info(f"\n{path_banner}")
                logging.info(f"USING VIDEO-ONLY PATH (modality={self.modality})")
                logging.info(f"Encoder output shape: {encoder_out.shape}, dtype: {encoder_out.dtype}")
                logging.info(f"{path_banner}")
            
            # Combined path (both audio and video)
            elif audio_features is not None and video_features is not None and self.modality == "both":
                # Determine a common sequence length for both features
                audio_seq_len = audio_features.size(1)
                video_seq_len = video_features.size(1)
                common_seq_len = min(audio_seq_len, video_seq_len, self.max_seq_len)
                
                # Truncate both to common length
                audio_features = audio_features[:, :common_seq_len, :]
                video_features = video_features[:, :common_seq_len, :]
                
                # Project each to LLM dimension separately
                logging.info(f"Audio features before connector: shape={audio_features.shape}, dtype={audio_features.dtype}")
                audio_llm_features = self.audio_connector(audio_features)
                logging.info(f"Audio features after connector: shape={audio_llm_features.shape}, dtype={audio_llm_features.dtype}")
                
                logging.info(f"Video features before connector: shape={video_features.shape}, dtype={video_features.dtype}")
                video_llm_features = self.video_connector(video_features)
                logging.info(f"Video features after connector: shape={video_llm_features.shape}, dtype={video_llm_features.dtype}")
                
                # Average both features (simple fusion)
                encoder_out = audio_llm_features * self.fusion_scale + video_llm_features * (1 - self.fusion_scale)
                batch_size, seq_len = encoder_out.size(0), encoder_out.size(1)
                path_banner = "=" * 80
                logging.info(f"\n{path_banner}")
                logging.info(f"USING COMBINED (AUDIO+VIDEO) PATH (modality={self.modality})")
                logging.info(f"Fusion scale: {self.fusion_scale}")
                logging.info(f"Encoder output shape: {encoder_out.shape}, dtype: {encoder_out.dtype}")
                logging.info(f"{path_banner}")
            
            else:
                raise ValueError("At least one of audio or video must be provided and match the specified modality")
            
            # Final NaN check before LLM
            if torch.isnan(encoder_out).any() or torch.isinf(encoder_out).any():
                logging.warning(f"NaN/Inf detected in encoder output before LLM")
                encoder_out = torch.nan_to_num(encoder_out)
            
            # Ensure encoder output matches LLM's expected dtype
            llm_input_dtype = next(self.llm.parameters()).dtype
            if encoder_out.dtype != llm_input_dtype:
                logging.info(f"Converting encoder output from {encoder_out.dtype} to {llm_input_dtype}")
                encoder_out = encoder_out.to(llm_input_dtype)
            
            # Create attention mask (all 1s)
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)
            
            # Handle prompt if provided
            if prompt is not None:
                embedding_layer = self.llm.get_input_embeddings()
                prompt_embeds = embedding_layer(prompt.to(self.device))
                encoder_out = torch.cat([prompt_embeds, encoder_out], dim=1)
                attention_mask = torch.ones((batch_size, encoder_out.size(1)), dtype=torch.long, device=self.device)
                
            # Prepare labels for loss computation if needed
            if return_loss and labels is not None:
                # Adjust labels to match sequence length
                if labels.size(1) != encoder_out.size(1):
                    if labels.size(1) < encoder_out.size(1):
                        # Pad labels with -100
                        padding = torch.full(
                            (batch_size, encoder_out.size(1) - labels.size(1)),
                            -100,  # ignored in loss
                            dtype=labels.dtype,
                            device=labels.device
                        )
                        labels = torch.cat([labels, padding], dim=1)
                    else:
                        # Truncate labels
                        labels = labels[:, :encoder_out.size(1)]
            
            # Forward pass through LLM
            if return_loss and labels is not None:
                # For training with loss calculation
                outputs = self.llm(
                    inputs_embeds=encoder_out,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
                return outputs
            else:
                # For inference/generation, return hidden states for later use
                return type('obj', (object,), {
                    'hidden_states': encoder_out,
                    'attention_mask': attention_mask
                })
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
            
            # We don't need to save the video_to_audio_dim connector anymore
            
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
            raise
        
        # Save config
        logging.info(f"Saving config to {output_dir}")
        try:
            config = {
                "audio_dim": self.audio_dim,
                "video_dim": self.video_dim,
                "llm_dim": self.llm_dim,
                "whisper_model": self.whisper.config._name_or_path,
                "clip_model": self.clip.config._name_or_path,
                "llm_path": self.tokenizer.vocab_size,
                "use_fp16": self.use_fp16,
                "use_lora": self.use_lora,
                "use_4bit": self.use_4bit,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "freeze_encoders": self.freeze_encoders,
                "freeze_llm": self.freeze_llm,
                "modality": self.modality,
                "max_seq_len": self.max_seq_len,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "model_version": "1.0.0",
                "save_timestamp": datetime.datetime.now().isoformat(),
            }
            
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving config: {e}")
            raise
        
        # Save the tokenizer
        logging.info(f"Saving tokenizer to {output_dir}")
        try:
            tokenizer_dir = os.path.join(output_dir, "tokenizer")
            os.makedirs(tokenizer_dir, exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_dir)
        except Exception as e:
            logging.error(f"Error saving tokenizer: {e}")
            raise
        
        # Save LLM and its components
        llm_output_dir = os.path.join(output_dir, "llm")
        os.makedirs(llm_output_dir, exist_ok=True)
        
        try:
            # Save LLM configuration
            if hasattr(self.llm, "config"):
                self.llm.config.save_pretrained(llm_output_dir)
            
            # Save LLM-specific files
            if hasattr(self.llm, "generation_config"):
                self.llm.generation_config.save_pretrained(llm_output_dir)
                
        except Exception as e:
            logging.error(f"Error saving LLM: {e}")
            raise
            
        # Verify the save
        try:
            self._verify_save(output_dir)
        except Exception as e:
            logging.error(f"Save verification failed: {e}")
            raise
        
        logging.info(f"Model successfully saved to {output_dir}")
        return output_dir
        
    def _verify_save(self, output_dir):
        """Verify that all necessary files were saved correctly"""
        required_files = [
            "audio_connector.pt",
            "video_connector.pt",
            "model_config.json",
            "config.json",
            os.path.join("tokenizer", "tokenizer_config.json"),
            os.path.join("llm", "config.json"),
        ]
        
        for file_path in required_files:
            full_path = os.path.join(output_dir, file_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Required file {file_path} not found in save directory")

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
        
        # The video_to_audio_dim connector is no longer needed
        
        logging.info(f"Model loaded from {model_dir}")
        model._verify_save(model_dir)
        
        return model

    def _load_whisper_model(self, whisper_model):
        """Load Whisper model and processor"""
        try:
            logging.info(f"Loading Whisper model: {whisper_model}")
            whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
            whisper = WhisperModel.from_pretrained(whisper_model).to(self.device)
            
            # Convert to specified dtype only if fp16 is being used
            if self.use_fp16:
                whisper = whisper.to(self.dtype)
                
            return whisper, whisper_processor
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            raise
    
    def _load_clip_model(self, clip_model):
        """Load CLIP model and processor"""
        try:
            logging.info(f"Loading CLIP model: {clip_model}")
            clip_processor = CLIPProcessor.from_pretrained(clip_model)
            clip = CLIPVisionModel.from_pretrained(clip_model).to(self.device)
            
            # Convert to specified dtype only if fp16 is being used
            if self.use_fp16:
                clip = clip.to(self.dtype)
                
            return clip, clip_processor
        except Exception as e:
            logging.error(f"Error loading CLIP model: {e}")
            raise
    
    def _freeze_model(self, model):
        """Freeze all parameters in a model"""
        for param in model.parameters():
            param.requires_grad = False
    
    def _load_llm(self, llm_path):
        """Load the language model with proper configuration"""
        try:
            # Set pad token BEFORE loading the model to ensure proper initialization
            if self.tokenizer.pad_token is None:
                logging.info("Tokenizer has no pad_token, setting pad_token to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # Add the pad token to ensure it's in the vocabulary
                special_tokens_dict = {'pad_token': self.tokenizer.eos_token}
                self.tokenizer.add_special_tokens(special_tokens_dict)
            
            # Load the model with the updated tokenizer, possibly with 4-bit quantization
            if self.use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    import bitsandbytes as bnb
                    
                    logging.info("Loading LLM in 4-bit quantization mode")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    llm = AutoModelForCausalLM.from_pretrained(
                        llm_path,
                        quantization_config=quantization_config,
                        device_map=self.device,
                    )
                except ImportError:
                    logging.warning("bitsandbytes not installed. Falling back to regular loading.")
                    llm = AutoModelForCausalLM.from_pretrained(
                        llm_path,
                        torch_dtype=self.dtype,
                        device_map=self.device,
                    )
            else:
                # Regular loading without quantization
                llm = AutoModelForCausalLM.from_pretrained(
                    llm_path,
                    torch_dtype=self.dtype,
                    device_map=self.device,
                )
            
            # Resize token embeddings to match the updated vocabulary
            if self.tokenizer.vocab_size != llm.config.vocab_size:
                logging.info(f"Resizing token embeddings from {llm.config.vocab_size} to {len(self.tokenizer)}")
                llm.resize_token_embeddings(len(self.tokenizer))
            
            # Apply LoRA to the LLM if requested
            if self.use_lora:
                try:
                    from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
                    
                    # Use more stable LoRA settings while still allowing learning
                    logging.info("Applying LoRA adapter with stable settings")
                    
                    # Prepare model for LoRA if needed
                    if self.use_4bit:
                        llm = prepare_model_for_kbit_training(llm)
                    
                    # Balanced settings for stability and learning capacity
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=8,  # Lower rank than default but still effective
                        lora_alpha=16,  # Balanced alpha
                        lora_dropout=0.05,  # Minimal dropout
                        # Target key components for effectiveness
                        target_modules=["q_proj", "v_proj"],
                        bias="none",  # Don't train biases for stability
                        # Better initialization
                        init_lora_weights="gaussian"  # Small gaussian init
                    )
                    
                    # Apply LoRA to the model
                    llm = get_peft_model(llm, peft_config)
                    
                    # Verify trainable parameters
                    trainable_params = sum(p.numel() for p in llm.parameters() if p.requires_grad)
                    all_params = sum(p.numel() for p in llm.parameters())
                    logging.info(f"LoRA trainable params: {trainable_params} ({trainable_params/all_params:.4%})")
                    
                except ImportError:
                    logging.warning("PEFT not installed. Running without LoRA adaptation.")
            
            # Freeze the LLM if requested
            if self.freeze_llm:
                logging.info("Freezing the entire LLM for stability")
                for param in llm.parameters():
                    param.requires_grad = False
                
                # Verify LLM is frozen
                trainable_params = sum(p.numel() for p in llm.parameters() if p.requires_grad)
                if trainable_params == 0:
                    logging.info("LLM successfully frozen - no trainable parameters")
                else:
                    logging.warning(f"LLM has {trainable_params} trainable parameters despite freezing attempt")
            
            return llm
        except Exception as e:
            logging.error(f"Error loading LLM: {e}")
            raise 

    def _log_parameter_count(self):
        """Count and log parameters by component"""
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
        logging.info("MODEL PARAMETER SUMMARY")
        logging.info("=" * 80)
        logging.info(f"{'Component':<20} {'Parameters':<15} {'Trainable':<15} {'% Trainable':<15}")
        logging.info("-" * 80)
        logging.info(f"{'Whisper':<20} {whisper_params:,d} {whisper_trainable:,d} {100*whisper_trainable/max(1, whisper_params):.2f}%")
        logging.info(f"{'CLIP':<20} {clip_params:,d} {clip_trainable:,d} {100*clip_trainable/max(1, clip_params):.2f}%")
        logging.info(f"{'LLM':<20} {llm_params:,d} {llm_trainable:,d} {100*llm_trainable/max(1, llm_params):.2f}%")
        logging.info(f"{'Audio Connector':<20} {audio_connector_params:,d} {audio_connector_trainable:,d} {100*audio_connector_trainable/max(1, audio_connector_params):.2f}%")
        logging.info(f"{'Video Connector':<20} {video_connector_params:,d} {video_connector_trainable:,d} {100*video_connector_trainable/max(1, video_connector_params):.2f}%")
        logging.info("-" * 80)
        logging.info(f"{'TOTAL':<20} {total_params:,d} {total_trainable:,d} {100*total_trainable/max(1, total_params):.2f}%")
        logging.info("=" * 80)
        
    def encode_audio(self, audio, attention_mask=None):
        """Encode audio using Whisper"""
        with torch.no_grad():
            try:
                # Ensure audio is on the correct device
                if audio.device != self.device:
                    audio = audio.to(self.device)
                
                # IMPORTANT: Convert to the SAME dtype as the Whisper model
                whisper_dtype = next(self.whisper.parameters()).dtype
                if audio.dtype != whisper_dtype:
                    audio = audio.to(whisper_dtype)
                
                # Check for NaN values
                if torch.isnan(audio).any() or torch.isinf(audio).any():
                    logging.warning("NaN/Inf values detected in audio input, replacing with zeros")
                    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Get Whisper features
                whisper_output = self.whisper.encoder(audio)
                
                # Get the last hidden state from the output
                if hasattr(whisper_output, "last_hidden_state"):
                    features = whisper_output.last_hidden_state
                else:
                    features = whisper_output
                
                # Log the dimensionality of the audio features
                logging.info(f"Audio encoder output shape: {features.shape}, dtype: {features.dtype}")
                
                return features
                
            except Exception as e:
                logging.error(f"Error encoding audio: {e}")
                logging.error(traceback.format_exc())
                
                # Return a dummy tensor with the expected shape and dtype
                # This helps training continue even if there's an issue with this example
                batch_size = audio.size(0)
                expected_seq_len = audio.size(1) // 16
                whisper_dtype = next(self.whisper.parameters()).dtype
                return torch.zeros(
                    (batch_size, expected_seq_len, self.audio_dim),
                    device=self.device,
                    dtype=whisper_dtype  # Match Whisper's dtype
                )
    
    def encode_video(self, video, attention_mask=None):
        """Encode video frames using CLIP"""
        with torch.no_grad():
            try:
                # Ensure video is on the correct device
                if video.device != self.device:
                    video = video.to(self.device)
                
                # IMPORTANT: Convert to the SAME dtype as the CLIP model
                clip_dtype = next(self.clip.parameters()).dtype
                if video.dtype != clip_dtype:
                    video = video.to(clip_dtype)
                
                # Check for NaN values
                if torch.isnan(video).any() or torch.isinf(video).any():
                    logging.warning("NaN/Inf values detected in video input, replacing with zeros")
                    video = torch.nan_to_num(video, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Get batch size and num_frames from input dimensions
                logging.info(f"Input video shape: {video.shape}")
                
                # Process video based on its shape
                if video.dim() == 5 and video.size(2) == 3:
                    # Input is [batch_size, frames, channels, height, width]
                    batch_size, num_frames = video.size(0), video.size(1)
                    # Reshape to [batch*frames, channels, height, width]
                    flat_video = video.reshape(-1, video.size(2), video.size(3), video.size(4))
                    logging.info(f"Reshaped video to: {flat_video.shape}")
                elif video.dim() == 4 and video.size(1) == 3:
                    # Input is [batch*frames, channels, height, width]
                    flat_video = video
                    batch_size = 1
                    num_frames = flat_video.size(0)
                    logging.info(f"Video already in CLIP format: {flat_video.shape}")
                else:
                    raise ValueError(f"Unexpected video shape: {video.shape}. Expected [batch, frames, 3, height, width] or [frames, 3, height, width]")
                
                # Process with CLIP model using the model's native dtype
                with torch.amp.autocast('cuda', enabled=False):
                    clip_output = self.clip(flat_video)
                    video_features = clip_output.last_hidden_state
                
                # Mean across sequence dimension to get single feature vector per frame
                video_features = video_features.mean(dim=1)
                
                # Reshape back to [batch_size, frames, features]
                video_features = video_features.reshape(batch_size, num_frames, -1)
                logging.info(f"Final video features shape: {video_features.shape}")
                
                # Log the dimensionality of the video features
                logging.info(f"Video encoder output shape: {video_features.shape}, dtype: {video_features.dtype}")
                
                return video_features
                
            except Exception as e:
                logging.error(f"Error in CLIP processing: {e}")
                logging.error(traceback.format_exc())
                
                # Create a proper fallback with the right dimensions
                if 'batch_size' not in locals():
                    if video.dim() == 5:
                        batch_size = video.size(0)
                        num_frames = video.size(1)
                    else:
                        batch_size = 1
                        num_frames = video.size(0) if video.dim() == 4 else 1
                
                clip_dtype = next(self.clip.parameters()).dtype
                return torch.zeros(
                    (batch_size, num_frames, self.video_dim),
                    device=self.device,
                    dtype=clip_dtype  # Match CLIP's dtype
                )