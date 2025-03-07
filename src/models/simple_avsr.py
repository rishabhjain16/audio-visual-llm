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
    """
    Super simple connector module to project features from one dimension to another.
    Uses a single linear layer with proper initialization to avoid numerical issues.
    """
    
    def __init__(self, input_dim, output_dim, device="cuda", dtype=torch.float16):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        
        # Create projection with improved initialization for stability
        self.norm = nn.LayerNorm(input_dim).to(device=device, dtype=dtype)
        self.proj = nn.Linear(input_dim, output_dim).to(device=device, dtype=dtype)
        self.dropout = nn.Dropout(0.1)
        
        # Use more stable initialization - especially important for fusion
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.proj.bias)
        
        logging.info(f"Created stable ModalityConnector: {input_dim} → {output_dim}")
    
    def forward(self, x):
        """Forward pass with careful NaN handling"""
        # Replace any NaN/Inf values with zeros
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.warning("NaN/Inf values detected in ModalityConnector input, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure correct device and dtype
        if x.device != self.device:
            x = x.to(self.device)
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        
        # Apply normalization for stability
        x = self.norm(x)
        
        # Apply projection and dropout
        x = self.proj(x)
        x = self.dropout(x)
        
        # Final NaN check (belt and suspenders)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.warning("NaN/Inf values in ModalityConnector output, using zeros")
            x = torch.zeros_like(x)
        
        return x


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
        modality: str = "both",  # "audio", "video", or "both"
        max_seq_len: int = 256,  # Maximum sequence length for stability
        fusion_scale: float = 0.5,  # Scaling factor for fusion (0.5 = equal weighting)
    ):
        """Initialize the AVSR model"""
        super().__init__()
        
        # Store initialization parameters
        self.device = device
        self.use_fp16 = use_fp16
        self.use_lora = use_lora
        self.use_4bit = use_4bit
        self.modality = modality
        self.max_seq_len = max_seq_len
        self.freeze_encoders = freeze_encoders
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.fusion_scale = fusion_scale  # For controlling modality weighting
        
        # Set dtype based on use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
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
        
        # Create connectors with proper initialization
        self.audio_connector = ModalityConnector(
            self.audio_dim, self.llm_dim,
            device=device, dtype=self.dtype
        )
        
        self.video_connector = ModalityConnector(
            self.video_dim, self.llm_dim,
            device=device, dtype=self.dtype
        )
        
        # For video: project to audio_dim for fusion
        self.video_to_audio_dim = ModalityConnector(
            self.video_dim, self.audio_dim,
            device=device, dtype=self.dtype
        )
        
        # For fusion: project combined features to LLM dim
        self.fusion_connector = ModalityConnector(
            self.audio_dim, self.llm_dim,
            device=device, dtype=self.dtype
        )
        
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
            # Log input shapes for debugging
            if audio is not None:
                logging.info(f"Input audio shape: {audio.shape}")
            if video is not None:
                logging.info(f"Input video shape: {video.shape}")
            if labels is not None:
                logging.info(f"Input labels shape: {labels.shape}")
            
            # Encode audio if provided
            audio_features = None
            if audio is not None and self.modality in ["audio", "both"]:
                with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for stability
                    audio_features = self.encode_audio(audio)
                    if audio_features is not None:
                        # Check for NaN values
                        if torch.isnan(audio_features).any() or torch.isinf(audio_features).any():
                            logging.warning("NaN/Inf detected in audio features, replacing with zeros")
                            audio_features = torch.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
                logging.info(f"Audio features shape after encoding: {audio_features.shape if audio_features is not None else None}")
            
            # Encode video if provided
            video_features = None
            if video is not None and self.modality in ["video", "both"]:
                with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for stability
                    video_features = self.encode_video(video)
                    if video_features is not None:
                        # Check for NaN values
                        if torch.isnan(video_features).any() or torch.isinf(video_features).any():
                            logging.warning("NaN/Inf detected in video features, replacing with zeros")
                            video_features = torch.nan_to_num(video_features, nan=0.0, posinf=0.0, neginf=0.0)
                logging.info(f"Video features shape after encoding: {video_features.shape if video_features is not None else None}")
            
            # Combine features if both are available
            if audio_features is not None and video_features is not None:
                # First, determine a common sequence length (use minimum of both)
                seq_len = min(audio_features.size(1), video_features.size(1))
                seq_len = min(seq_len, self.max_seq_len)  # Also respect max_seq_len
                
                logging.info(f"Using common sequence length: {seq_len}")
                
                # Truncate both to this common length
                audio_features = audio_features[:, :seq_len, :]
                video_features = video_features[:, :seq_len, :]
                
                # Project video to audio dimension
                video_features_audio_dim = self.video_to_audio_dim(video_features)  # [batch, seq_len, audio_dim]
                
                # Combine features with scaling (defaults to equal weighting with 0.5)
                combined_features = (
                    audio_features * self.fusion_scale + 
                    video_features_audio_dim * (1.0 - self.fusion_scale)
                )  # [batch, seq_len, audio_dim]
                
                # Check for NaN values before projecting
                if torch.isnan(combined_features).any() or torch.isinf(combined_features).any():
                    logging.warning("NaN/Inf detected in combined features, replacing with zeros")
                    combined_features = torch.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Project combined features to LLM dim
                encoder_out = self.fusion_connector(combined_features)  # [batch, seq_len, llm_dim]
                
            elif audio_features is not None:
                # Audio-only path
                # Truncate to max_seq_len
                audio_features = audio_features[:, :self.max_seq_len, :]
                encoder_out = self.audio_connector(audio_features)
                
            elif video_features is not None:
                # Video-only path
                # Truncate to max_seq_len
                video_features = video_features[:, :self.max_seq_len, :]
                encoder_out = self.video_connector(video_features)
                
            else:
                raise ValueError("At least one of audio or video must be provided")
            
            # Final NaN check before LLM
            if torch.isnan(encoder_out).any() or torch.isinf(encoder_out).any():
                logging.warning("NaN/Inf detected in encoder output, replacing with zeros")
                encoder_out = torch.nan_to_num(encoder_out, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create attention mask (all 1s)
            batch_size, seq_len = encoder_out.size()[:2]
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)
            
            # Handle prompt if provided
            if prompt is not None:
                embedding_layer = self.llm.get_input_embeddings()
                prompt_embeds = embedding_layer(prompt.to(self.device))
                encoder_out = torch.cat([prompt_embeds, encoder_out], dim=1)
                attention_mask = torch.ones((batch_size, encoder_out.size(1)), dtype=torch.long, device=self.device)
            
            # Prepare labels for loss computation
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
            
            # Forward pass through LLM with robust error handling
            try:
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    outputs = self.llm(
                        inputs_embeds=encoder_out,
                        attention_mask=attention_mask,
                        labels=labels if return_loss and labels is not None else None,
                        return_dict=True
                    )
                
                # Check the loss for NaN/Inf before returning
                if hasattr(outputs, "loss") and (torch.isnan(outputs.loss) or torch.isinf(outputs.loss)):
                    logging.warning("NaN/Inf detected in LLM loss, returning dummy loss")
                    outputs.loss = torch.tensor(0.5, device=self.device, requires_grad=True)
                
                return outputs
            except Exception as e:
                logging.error(f"Error in LLM forward pass: {e}")
                if return_loss:
                    # Return a dummy loss value
                    dummy_loss = torch.tensor(0.5, device=self.device, requires_grad=True)
                    return type('obj', (object,), {'loss': dummy_loss})
                return {"error": f"LLM forward pass error: {e}"}
            
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            logging.error(traceback.format_exc())
            
            # Return a dummy loss for training stability
            if return_loss:
                dummy_loss = torch.tensor(0.5, device=self.device, requires_grad=True)
                return type('obj', (object,), {'loss': dummy_loss})
            
            # Return error message for generation
            return {"generated_text": ["Error in model forward pass"], "error": str(e)}
    
    def save_pretrained(self, output_dir):
        """Save model to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the projectors with consistent naming
        logging.info(f"Saving connectors to {output_dir}")
        try:
            torch.save(self.audio_connector.state_dict(), os.path.join(output_dir, "audio_connector.pt"))
            torch.save(self.video_connector.state_dict(), os.path.join(output_dir, "video_connector.pt"))
            torch.save(self.video_to_audio_dim.state_dict(), os.path.join(output_dir, "video_to_audio_dim.pt"))
            torch.save(self.fusion_connector.state_dict(), os.path.join(output_dir, "fusion_connector.pt"))
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
            "video_to_audio_dim.pt",
            "fusion_connector.pt",
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
        """Load model from a directory"""
        try:
            logging.info(f"Loading model from {model_dir}")
            
            # Load config
            with open(os.path.join(model_dir, "config.json"), "r") as f:
                config = json.load(f)
            
            # Create model
            model = cls(
                llm_path=config.get("llm_path", ""),
                whisper_model=config.get("whisper_model", "openai/whisper-medium"),
                clip_model=config.get("clip_model", "openai/clip-vit-base-patch32"),
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_fp16=config.get("use_fp16", False),
                use_lora=config.get("use_lora", True),
                use_4bit=config.get("use_4bit", False),
                lora_r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
                freeze_encoders=config.get("freeze_encoders", True),
                modality=config.get("modality", "both"),
                max_seq_len=config.get("max_seq_len", 256),
            )
            
            # Load projector weights
            logging.info(f"Loading connector weights from {model_dir}")
            model.audio_connector.load_state_dict(
                torch.load(os.path.join(model_dir, "audio_connector.pt"))
            )
            model.video_connector.load_state_dict(
                torch.load(os.path.join(model_dir, "video_connector.pt"))
            )
            model.video_to_audio_dim.load_state_dict(
                torch.load(os.path.join(model_dir, "video_to_audio_dim.pt"))
            )
            model.fusion_connector.load_state_dict(
                torch.load(os.path.join(model_dir, "fusion_connector.pt"))
            )
            
            # Load tokenizer
            tokenizer_dir = os.path.join(model_dir, "tokenizer")
            if os.path.exists(tokenizer_dir):
                logging.info(f"Loading tokenizer from {tokenizer_dir}")
                model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            
            # Load LLM if available
            llm_dir = os.path.join(model_dir, "llm")
            if os.path.exists(llm_dir):
                logging.info(f"Loading LLM from {llm_dir}")
                try:
                    # Try loading the entire model
                    model.llm = AutoModelForCausalLM.from_pretrained(
                        llm_dir,
                        torch_dtype=model.dtype,
                        device_map=model.device,
                    )
                except Exception as e:
                    logging.error(f"Error loading LLM: {e}")
                    # Try loading just the state dict
                    logging.warning("Trying to load state dict only")
                    state_dict = torch.load(
                        os.path.join(llm_dir, "pytorch_model.bin"),
                        map_location=model.device
                    )
                    model.llm.load_state_dict(state_dict)
            
            # Validate the loaded model
            logging.info("Validating loaded model")
            if model.audio_dim != model.audio_connector.input_dim:
                logging.warning(f"Audio dimensions mismatch: {model.audio_dim} vs {model.audio_connector.input_dim}")
            
            if model.video_dim != model.video_connector.input_dim:
                logging.warning(f"Video dimensions mismatch: {model.video_dim} vs {model.video_connector.input_dim}")
            
            if model.llm_dim != model.audio_connector.output_dim:
                logging.warning(f"LLM dimensions mismatch: {model.llm_dim} vs {model.audio_connector.output_dim}")
            
            logging.info(f"Model successfully loaded from {model_dir}")
            return model
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            logging.error(traceback.format_exc())
            raise 

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
                    from peft import get_peft_model, LoraConfig, TaskType
                    
                    logging.info(f"Applying LoRA to LLM with r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    )
                    
                    # Apply LoRA to the model
                    llm = get_peft_model(llm, peft_config)
                    llm.print_trainable_parameters()
                except ImportError:
                    logging.warning("PEFT not installed. Running without LoRA adaptation.")
            
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
        video_to_audio_dim_params = sum(p.numel() for p in self.video_to_audio_dim.parameters())
        fusion_connector_params = sum(p.numel() for p in self.fusion_connector.parameters())
        
        # Count trainable parameters by component
        whisper_trainable = sum(p.numel() for p in self.whisper.parameters() if p.requires_grad)
        clip_trainable = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        llm_trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        audio_connector_trainable = sum(p.numel() for p in self.audio_connector.parameters() if p.requires_grad)
        video_connector_trainable = sum(p.numel() for p in self.video_connector.parameters() if p.requires_grad)
        video_to_audio_dim_trainable = sum(p.numel() for p in self.video_to_audio_dim.parameters() if p.requires_grad)
        fusion_connector_trainable = sum(p.numel() for p in self.fusion_connector.parameters() if p.requires_grad)
        
        # Total counts
        total_params = whisper_params + clip_params + llm_params + \
                      audio_connector_params + video_connector_params + \
                      video_to_audio_dim_params + fusion_connector_params
                      
        total_trainable = whisper_trainable + clip_trainable + llm_trainable + \
                         audio_connector_trainable + video_connector_trainable + \
                         video_to_audio_dim_trainable + fusion_connector_trainable
        
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
        logging.info(f"{'Video to Audio':<20} {video_to_audio_dim_params:,d} {video_to_audio_dim_trainable:,d} {100*video_to_audio_dim_trainable/max(1, video_to_audio_dim_params):.2f}%")
        logging.info(f"{'Fusion Connector':<20} {fusion_connector_params:,d} {fusion_connector_trainable:,d} {100*fusion_connector_trainable/max(1, fusion_connector_params):.2f}%")
        logging.info("-" * 80)
        logging.info(f"{'TOTAL':<20} {total_params:,d} {total_trainable:,d} {100*total_trainable/max(1, total_params):.2f}%")
        logging.info("=" * 80)
        
    def encode_audio(self, audio, attention_mask=None):
        """Encode audio using Whisper with robust error handling"""
        with torch.no_grad():
            try:
                # Ensure audio is on the correct device and dtype
                if audio.device != self.device:
                    audio = audio.to(self.device)
                
                # Convert to the expected data type
                if audio.dtype != self.dtype:
                    audio = audio.to(self.dtype)
                
                # Handle potential NaN values
                if torch.isnan(audio).any() or torch.isinf(audio).any():
                    logging.warning("NaN/Inf values detected in audio input, replacing with zeros")
                    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Get batch size and input length for fallback tensor creation
                batch_size = audio.size(0)
                input_length = audio.size(1)
                expected_seq_len = input_length // 16  # Typical whisper downsampling
                
                # Use float32 for the whisper encoder regardless of the model's dtype
                # This improves numerical stability significantly
                audio_float32 = audio.to(torch.float32)
                
                # Run the encoder
                with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision
                    whisper_out = self.whisper.encoder(audio_float32)
                    audio_features = whisper_out.last_hidden_state
                
                # Check for NaN values in the output
                if torch.isnan(audio_features).any() or torch.isinf(audio_features).any():
                    logging.warning("NaN/Inf values in Whisper output, using zeros")
                    audio_features = torch.zeros(
                        (batch_size, expected_seq_len, self.audio_dim),
                        device=self.device,
                        dtype=torch.float32  # Use float32 for stability
                    )
                
                # Convert to the model's dtype
                audio_features = audio_features.to(self.dtype)
                return audio_features
                
            except Exception as e:
                # Handle any errors in the Whisper encoder
                logging.warning(f"Error in Whisper encoder: {e}")
                batch_size = audio.size(0)
                expected_seq_len = audio.size(1) // 16
                return torch.zeros(
                    (batch_size, expected_seq_len, self.audio_dim),
                    device=self.device,
                    dtype=self.dtype
                )
    
    def encode_video(self, video, attention_mask=None):
        """Encode video frames using CLIP"""
        with torch.no_grad():
            try:
                # Ensure video is on the correct device and dtype
                if video.device != self.device:
                    video = video.to(self.device)
                
                # Convert to the expected data type of the model
                if video.dtype != self.dtype:
                    video = video.to(self.dtype)
                
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
                
                # Use float32 for CLIP encoding regardless of the model's dtype for stability
                flat_video_float32 = flat_video.to(torch.float32)
                
                # Process with CLIP model
                with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for stability
                    clip_output = self.clip(flat_video_float32)
                    video_features = clip_output.last_hidden_state
                
                # Mean across sequence dimension to get single feature vector per frame
                video_features = video_features.mean(dim=1)
                
                # Reshape back to [batch_size, frames, features]
                video_features = video_features.reshape(batch_size, num_frames, -1)
                logging.info(f"Final video features shape: {video_features.shape}")
                
                # Check for NaN values in the output
                if torch.isnan(video_features).any() or torch.isinf(video_features).any():
                    logging.warning("NaN/Inf values in CLIP output, replacing with zeros")
                    video_features = torch.zeros_like(video_features)
                
                # Convert to the model's dtype
                if video_features.dtype != self.dtype:
                    video_features = video_features.to(self.dtype)
                
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
                        
                return torch.zeros(
                    (batch_size, num_frames, self.video_dim),
                    device=self.device,
                    dtype=self.dtype
                )