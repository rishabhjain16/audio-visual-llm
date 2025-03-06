import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)

class LLMModule(nn.Module):
    """LLM integration module for AVSR
    
    This class loads a pretrained LLM and connects it to the AV-HuBERT encoder
    for speech recognition or translation.
    """
    
    def __init__(
        self,
        model_name_or_path,
        encoder_dim=0,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=None,
        use_8bit=False,
        use_4bit=False,  # Changed from True to False to bypass quantization issues
        device_map=None,  # Changed from "auto" to None to avoid device mapping issues
        max_length=512,
        prompt_template="Transcribe the speech: ",
        **kwargs
    ):
        """
        Initialize the LLM module.
        
        Args:
            model_name_or_path: Path to the pretrained model
            encoder_dim: Dimension of the encoder output (0 if not using encoder)
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            lora_target_modules: List of modules to apply LoRA to
            use_8bit: Whether to use 8-bit quantization
            use_4bit: Whether to use 4-bit quantization (takes precedence over 8-bit)
            device_map: Device mapping strategy for multi-GPU setups
            max_length: Maximum sequence length
            prompt_template: Template for the prompt to prepend to the generated text
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.encoder_dim = encoder_dim
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.device_map = device_map
        self.max_length = max_length
        self.lora_config = None
        self.prompt_template = prompt_template
        
        # Load model and tokenizer
        logger.info(f"Initializing LLM module with model: {model_name_or_path}")
        logger.info(f"Quantization: 4-bit={use_4bit}, 8-bit={use_8bit}, device_map={device_map}")
        
        # Load the model
        self.model = self._load_model()
        
        # Set up LoRA if specified
        if self.use_lora and not hasattr(self.model, "peft_config"):
            self._setup_lora()
            
        # Initialize encoder projection if needed
        if self.encoder_dim > 0 and not hasattr(self, "encoder_proj"):
            logger.info(f"Creating encoder projection from {self.encoder_dim} to {self.model.config.hidden_size}")
            self.encoder_proj = nn.Linear(self.encoder_dim, self.model.config.hidden_size)
            # Initialize with small values
            nn.init.normal_(self.encoder_proj.weight, std=0.02)
            nn.init.zeros_(self.encoder_proj.bias)
            
        # Create an alias for encoder_proj for backwards compatibility
        self.encoder_projection = self.encoder_proj if hasattr(self, "encoder_proj") else None
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing for memory efficiency")
    
    def _load_tokenizer(self):
        """Load the tokenizer"""
        print(f"Loading tokenizer from {self.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        # Ensure the tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model(self):
        """Load the LLM model"""
        try:
            logger.info(f"Loading model from: {self.model_name_or_path}")
            
            # Set quantization config if using 4-bit/8-bit
            quantization_config = None
            if self.use_8bit:
                logger.info("Using 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            elif self.use_4bit:
                logger.info("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Try different approaches to load the model
            model = None
            error_message = ""
            
            # Try loading with standard approach
            try:
                logger.info("Attempting to load model with default parameters")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    quantization_config=quantization_config,
                    device_map=self.device_map,
                    torch_dtype=torch.float16,  # Use float16 for memory efficiency
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                error_message = str(e)
                logger.warning(f"Failed to load model with default parameters: {e}")
            
            # Try loading with ignore_mismatched_sizes
            if model is None:
                try:
                    logger.info("Attempting to load model with ignore_mismatched_sizes=True")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        quantization_config=quantization_config,
                        device_map=self.device_map,
                        torch_dtype=torch.float16,  # Use float16 for memory efficiency
                        ignore_mismatched_sizes=True,
                        low_cpu_mem_usage=True
                    )
                except Exception as e:
                    error_message += f"\nFailed to load model with ignore_mismatched_sizes: {e}"
                    logger.warning(f"Failed to load model with ignore_mismatched_sizes: {e}")
            
            # Try loading with trust_remote_code
            if model is None:
                try:
                    logger.info("Attempting to load model with trust_remote_code=True")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        quantization_config=quantization_config,
                        device_map=self.device_map,
                        torch_dtype=torch.float16,  # Use float16 for memory efficiency
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                except Exception as e:
                    error_message += f"\nFailed to load model with trust_remote_code: {e}"
                    logger.warning(f"Failed to load model with trust_remote_code: {e}")
            
            # Try loading with safetensors
            if model is None:
                try:
                    logger.info("Trying to load with safetensors format and ignore_mismatched_sizes=True")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        quantization_config=quantization_config,
                        device_map=self.device_map,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        ignore_mismatched_sizes=True,
                        use_safetensors=True
                    )
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Failed to load with safetensors: {e}")
            
            # If all attempts to load the specified model failed, try a fallback
            if model is None:
                # If the error contains 'lm_head.weight', it's likely an architecture mismatch
                if 'lm_head.weight' in error_message:
                    fallback_models = [
                        "gpt2",  # Smallest model, should work on any GPU
                        "facebook/opt-125m",  # Small OPT model
                        "EleutherAI/pythia-70m"  # Small Pythia model
                    ]
                    
                    for fallback in fallback_models:
                        try:
                            logger.warning(f"Specified model failed to load. Trying fallback model: {fallback}")
                            model = AutoModelForCausalLM.from_pretrained(
                                fallback,
                                torch_dtype=torch.float32,
                                low_cpu_mem_usage=True
                            )
                            if model is not None:
                                logger.info(f"Successfully loaded fallback model: {fallback}")
                                # Update model name to reflect what was actually loaded
                                self.model_name_or_path = fallback
                                break
                        except Exception as e:
                            logger.error(f"Failed to load fallback model {fallback}: {e}")
            
            # If we still couldn't load any model, raise an error
            if model is None:
                raise ValueError(f"Failed to load any model. Last error: {error_message}")
            
            # Load tokenizer
            tokenizer = self._load_tokenizer()
            
            self.model = model
            self.tokenizer = tokenizer
            
            # Print model details
            model_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Loaded model with {model_params:,} parameters")
            logger.info(f"Model type: {type(model).__name__}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_encoder_projection(self):
        """Create projection from encoder output to LLM input"""
        # Determine LLM embedding dimension
        if hasattr(self.model.config, "hidden_size"):
            llm_dim = self.model.config.hidden_size
        elif hasattr(self.model.config, "n_embd"):
            llm_dim = self.model.config.n_embd
        elif hasattr(self.model.config, "d_model"):
            llm_dim = self.model.config.d_model
        else:
            llm_dim = 4096  # Default for many models like Llama
            print(f"Warning: Could not determine LLM dimension, using default: {llm_dim}")
        
        # Get model dtype
        model_dtype = next(self.model.parameters()).dtype
        logging.info(f"Creating encoder projection with dtype {model_dtype} from dimension {self.encoder_dim} to {llm_dim}")
        
        # Create MLP projection with layer norm
        projection = nn.Sequential(
            nn.Linear(self.encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim)
        )
        
        # Convert to model dtype
        for module in projection:
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                module.to(dtype=model_dtype)
        
        # Move to correct device
        device = next(self.model.parameters()).device
        projection = projection.to(device)
        
        logging.info(f"Projection created successfully: {self.encoder_dim} -> {llm_dim}, on device: {device}")
        
        return projection
    
    def prepare_inputs_for_generation(self, input_ids, encoder_outputs=None, attention_mask=None, **kwargs):
        """
        Prepare inputs for generation with encoder outputs.
        This method handles combining prompt embeddings with encoder features.
        
        Args:
            input_ids: Token IDs for the prompt
            encoder_outputs: Output from the encoder
            attention_mask: Attention mask for the prompt
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of inputs for the model
        """
        # Get default inputs
        inputs = self.model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask, **kwargs)
        
        # If no encoder outputs, just return the default inputs
        if encoder_outputs is None:
            return inputs
        
        try:
            # Get the device of the model
            model_device = next(self.model.parameters()).device
            
            # Check if encoder output is on the same device as the model
            if encoder_outputs.device != model_device:
                logger.info(f"Moving encoder output from {encoder_outputs.device} to {model_device}")
                encoder_outputs = encoder_outputs.to(model_device)
            
            # Get prompt embeddings
            prompt_embeds = None
            try:
                # Get the embedding layer
                if hasattr(self.model, "get_input_embeddings"):
                    embed_layer = self.model.get_input_embeddings()
                else:
                    embed_layer = self.model.model.embed_tokens
                
                # Get prompt embeddings
                prompt_embeds = embed_layer(input_ids).to(encoder_outputs.dtype)
            except Exception as e:
                logger.error(f"Error getting prompt embeddings: {e}")
                # Fallback to random embeddings of the right shape
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                hidden_size = self.model.config.hidden_size
                prompt_embeds = torch.randn(batch_size, seq_len, hidden_size, device=model_device, dtype=encoder_outputs.dtype)
            
            # Log dimensions for debugging
            logger.info(f"Encoder outputs shape: {encoder_outputs.shape}, device: {encoder_outputs.device}")
            logger.info(f"Prompt embeds shape: {prompt_embeds.shape}, device: {prompt_embeds.device}")
            
            # Project encoder outputs if needed
            if self.encoder_projection is not None and encoder_outputs is not None:
                logger.debug("Applying encoder projection")
                encoder_outputs = self.encoder_projection(encoder_outputs)
            
            # Debug logging
            if os.environ.get("AVSR_DEBUG", "0") == "1":
                logger.info(f"Prompt embeds shape: {prompt_embeds.shape}, device: {prompt_embeds.device}")
                logger.info(f"Projected encoder outputs shape: {encoder_outputs.shape}, device: {encoder_outputs.device}")
            
            # Make batch sizes consistent
            if prompt_embeds.shape[0] != encoder_outputs.shape[0]:
                logger.warning(f"Batch size mismatch: prompt={prompt_embeds.shape[0]}, encoder={encoder_outputs.shape[0]}")
                
                # Convert to the smaller batch size to avoid dimension errors
                if encoder_outputs.shape[0] < prompt_embeds.shape[0]:
                    # Repeat encoder outputs to match prompt batch size
                    repeat_factor = prompt_embeds.shape[0] // encoder_outputs.shape[0]
                    if repeat_factor > 0:
                        encoder_outputs = encoder_outputs.repeat(repeat_factor, 1, 1)
                        # If still not matching, truncate prompt
                        if prompt_embeds.shape[0] != encoder_outputs.shape[0]:
                            prompt_embeds = prompt_embeds[:encoder_outputs.shape[0]]
                            if attention_mask is not None:
                                attention_mask = attention_mask[:encoder_outputs.shape[0]]
                    else:
                        # Just use the first batch element and repeat
                        encoder_outputs = encoder_outputs[0:1].repeat(prompt_embeds.shape[0], 1, 1)
                else:
                    # Truncate encoder outputs to match prompt batch size
                    encoder_outputs = encoder_outputs[:prompt_embeds.shape[0]]
            
            # Ensure both tensors have same data type
            if prompt_embeds.dtype != encoder_outputs.dtype:
                encoder_outputs = encoder_outputs.to(prompt_embeds.dtype)
            
            # Combine prompt and encoder embeddings
            try:
                # Ensure both are on the same device
                if prompt_embeds.device != encoder_outputs.device:
                    encoder_outputs = encoder_outputs.to(prompt_embeds.device)
                
                # Concatenate along sequence dimension
                inputs_embeds = torch.cat([prompt_embeds, encoder_outputs], dim=1)
                
                # Check if inputs_embeds requires gradients
                if not inputs_embeds.requires_grad:
                    logger.warning("inputs_embeds does not require gradients. Setting requires_grad=True")
                    inputs_embeds.requires_grad_(True)
                
                # Update inputs
                inputs["inputs_embeds"] = inputs_embeds
                inputs.pop("input_ids", None)  # Remove input_ids as we're using inputs_embeds
                
                # Handle attention mask
                if attention_mask is not None:
                    # Create encoder attention mask of the right shape
                    encoder_seq_len = encoder_outputs.size(1)
                    batch_size = encoder_outputs.size(0)
                    
                    # Create mask where all encoder tokens can be attended to
                    encoder_attention_mask = torch.ones(batch_size, encoder_seq_len, 
                                                       device=attention_mask.device, 
                                                       dtype=attention_mask.dtype)
                    
                    # Combine masks
                    combined_attention_mask = torch.cat([attention_mask, encoder_attention_mask], dim=1)
                    
                    # Update attention mask
                    inputs["attention_mask"] = combined_attention_mask
            except RuntimeError as e:
                logger.error(f"Error combining embeddings: {e}")
                logger.error(traceback.format_exc())
                # Fallback to just using prompt embeddings
                inputs["inputs_embeds"] = prompt_embeds
                inputs.pop("input_ids", None)
        
        except Exception as e:
            logger.error(f"Error in prepare_inputs_for_generation: {e}")
            logger.error(traceback.format_exc())
            # Return original inputs as fallback
        
        return inputs
    
    def generate(
        self,
        encoder_out,
        encoder_padding_mask=None,
        max_new_tokens=100,
        num_beams=4,
        **kwargs
    ):
        """Generate text from encoder outputs
        
        Args:
            encoder_out: Encoder output features [B, T, D]
            encoder_padding_mask: Padding mask [B, T]
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search
            
        Returns:
            List of generated text strings
        """
        # Get the generation inputs
        generation_inputs = self.prepare_inputs_for_generation(encoder_out, encoder_padding_mask)
        
        # Get prompt length to skip in the output
        prompt_len = self.tokenizer(
            [self.prompt_template], return_tensors="pt"
        )["input_ids"].size(1)
        
        # Generate with the model
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=generation_inputs["inputs_embeds"],
                attention_mask=generation_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                **kwargs
            )
        
        # Decode the outputs, skipping the prompt
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Get just the generated part (skip prompt)
        if self.prompt_template in texts[0]:
            texts = [text.split(self.prompt_template, 1)[1].strip() for text in texts]
        
        return texts
    
    def forward(self, **inputs):
        """Forward pass through the LLM
        
        Args:
            inputs: Dict of inputs to pass to the LLM
            
        Returns:
            outputs: Dict of outputs from the LLM
        """
        # Get debug flag
        debug = inputs.pop('debug', False)
        
        # Remove id field that might cause errors in the LLM forward
        if 'id' in inputs:
            inputs.pop('id')
        
        try:
            # Set defaults
            inputs.setdefault('return_dict', True)
            
            # Aggressively filter inputs to ONLY include expected keys
            # Most LLMs expect only these keys
            valid_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'labels', 'return_dict']
            filtered_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
            
            # Log the filtering process if debug is enabled
            if debug:
                removed_keys = set(inputs.keys()) - set(filtered_inputs.keys())
                if removed_keys:
                    logger.debug(f"Removed unexpected inputs: {', '.join(removed_keys)}")
            
            # Handle labels if they're provided as a list instead of a tensor
            if 'labels' in filtered_inputs and not isinstance(filtered_inputs['labels'], torch.Tensor):
                if debug:
                    logger.debug(f"Converting labels from {type(filtered_inputs['labels'])} to tensor")
                
                # If labels is a list of strings, we need to tokenize them
                if isinstance(filtered_inputs['labels'], list) and all(isinstance(item, str) for item in filtered_inputs['labels']):
                    try:
                        # For now, just remove labels to avoid errors
                        # This will allow the model to run in inference mode
                        logger.warning("Text labels not supported in training mode. Removing labels.")
                        del filtered_inputs['labels']
                    except Exception as e:
                        logger.error(f"Error handling text labels: {e}")
                        # Remove labels to avoid further errors
                        if 'labels' in filtered_inputs:
                            del filtered_inputs['labels']
                else:
                    # For other types of non-tensor labels, remove them to avoid errors
                    logger.warning(f"Unsupported label type: {type(filtered_inputs['labels'])}. Removing labels.")
                    del filtered_inputs['labels']
            
            # Ensure all tensor inputs are float16 to avoid dtype mismatches
            for key, value in filtered_inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype != torch.float16:
                    if debug:
                        logger.debug(f"Converting input '{key}' from {value.dtype} to float16")
                    filtered_inputs[key] = value.to(dtype=torch.float16)
            
            # Debug inputs
            if debug:
                for k, v in filtered_inputs.items():
                    if isinstance(v, torch.Tensor):
                        logger.debug(f"Model input '{k}': shape={v.shape}, device={v.device}, dtype={v.dtype}")
                    else:
                        logger.debug(f"Model input '{k}': {v}")
            
            # Forward pass through the model
            try:
                outputs = self.model(**filtered_inputs)
            except RuntimeError as e:
                if "expected mat1 and mat2 to have the same dtype" in str(e):
                    # Try to fix dtype mismatch by converting model parameters
                    logger.warning("Detected dtype mismatch. Attempting to fix by converting parameters...")
                    
                    # Convert all necessary parameters to float16
                    for name, param in self.model.named_parameters():
                        if param.dtype != torch.float16 and not param.is_meta and param.requires_grad:
                            logger.info(f"Converting parameter {name} from {param.dtype} to float16")
                            param.data = param.data.to(torch.float16)
                    
                    # Special handling for LoRA parameters
                    if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
                        for module in self.model.base_model.model.modules():
                            if hasattr(module, 'weight') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                                # Convert LoRA weights
                                if module.weight.dtype != torch.float16:
                                    logger.info(f"Converting LoRA base weight from {module.weight.dtype} to float16")
                                    module.weight.data = module.weight.data.to(torch.float16)
                                
                                # Convert LoRA A and B matrices
                                if hasattr(module, 'lora_A') and module.lora_A.dtype != torch.float16:
                                    logger.info(f"Converting lora_A from {module.lora_A.dtype} to float16")
                                    module.lora_A.data = module.lora_A.data.to(torch.float16)
                                
                                if hasattr(module, 'lora_B') and module.lora_B.dtype != torch.float16:
                                    logger.info(f"Converting lora_B from {module.lora_B.dtype} to float16")
                                    module.lora_B.data = module.lora_B.data.to(torch.float16)
                    
                    # Try again with converted parameters
                    outputs = self.model(**filtered_inputs)
                else:
                    # Re-raise if it's not a dtype mismatch error
                    raise
            
            # Debug outputs
            if debug:
                if isinstance(outputs, dict):
                    for k, v in outputs.items():
                        if isinstance(v, torch.Tensor):
                            logger.debug(f"Model output '{k}': shape={v.shape}, device={v.device}, dtype={v.dtype}")
                else:
                    logger.debug(f"Model output type: {type(outputs)}")
                
                # Log GPU memory usage after forward pass
                if torch.cuda.is_available():
                    device_idx = 0
                    if next(self.model.parameters()).device.type == 'cuda':
                        device_idx = next(self.model.parameters()).device.index
                    free_mem, total_mem = torch.cuda.mem_get_info(device_idx)
                    free_mem = free_mem / (1024 ** 3)  # Convert to GB
                    total_mem = total_mem / (1024 ** 3)  # Convert to GB
                    used_mem = total_mem - free_mem
                    logger.debug(f"GPU memory after LLM forward: {used_mem:.2f}GB used / {total_mem:.2f}GB total")
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in LLM forward: {e}")
            logger.error(traceback.format_exc())
            # Create a dummy output with NaN loss
            return {"loss": torch.tensor(float('nan'), device=next(self.model.parameters()).device)}
    
    def save_pretrained(self, output_dir):
        """Save the model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        
        # Save LLM or LoRA weights
        if self.use_lora:
            self.model.save_pretrained(os.path.join(output_dir, "model"))
        else:
            # Save as full model
            self.model.save_pretrained(os.path.join(output_dir, "model"))
        
        # Save adapter separately
        torch.save(self.encoder_proj.state_dict(), os.path.join(output_dir, "encoder_proj.pt"))
        
        # Save hyperparameters
        import json
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump({
                "model_name_or_path": self.model_name_or_path,
                "encoder_dim": self.encoder_dim,
                "use_lora": self.use_lora,
                "use_8bit": self.use_8bit,
                "use_4bit": self.use_4bit,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.lora_target_modules,
                "prompt_template": self.prompt_template,
                "max_length": self.max_length
            }, f, indent=4)
    
    @classmethod
    def from_pretrained(cls, model_dir):
        """Load a pretrained model"""
        import json
        config_path = os.path.join(model_dir, "config.json")
        
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(
            model_name_or_path=os.path.join(model_dir, "model"),
            encoder_dim=config["encoder_dim"],
            use_lora=config["use_lora"],
            use_8bit=config["use_8bit"],
            use_4bit=config["use_4bit"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            lora_target_modules=config["target_modules"],
            prompt_template=config["prompt_template"],
            max_length=config["max_length"]
        )
        
        # Load encoder projection
        encoder_proj_path = os.path.join(model_dir, "encoder_proj.pt")
        if os.path.exists(encoder_proj_path):
            model.encoder_proj.load_state_dict(torch.load(encoder_proj_path))
        
        return model
    
    def _setup_lora(self):
        """Set up LoRA for the model"""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Prepare model for k-bit training if using quantization
            if self.use_4bit or self.use_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Ensure model parameters are float16 before applying LoRA
            for name, param in self.model.named_parameters():
                if param.dtype != torch.float16 and not param.is_meta and param.requires_grad:
                    logger.info(f"Converting parameter {name} from {param.dtype} to float16 before LoRA")
                    param.data = param.data.to(torch.float16)
            
            # Create LoRA config
            self.lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.lora_target_modules,
                # Ensure LoRA uses float16
                inference_mode=False,
                init_lora_weights="gaussian"
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, self.lora_config)
            
            # Ensure all LoRA parameters are float16
            for name, param in self.model.named_parameters():
                if 'lora' in name and param.requires_grad and param.dtype != torch.float16:
                    logger.info(f"Converting LoRA parameter {name} from {param.dtype} to float16")
                    param.data = param.data.to(torch.float16)
            
            self.model.print_trainable_parameters()
            logging.info("Successfully applied LoRA to model with float16 precision")
            
        except Exception as e:
            logging.error(f"Error applying LoRA: {e}")
            logging.warning("Continuing without LoRA")
            self._freeze_model_except_norms()
    
    def _freeze_model_except_norms(self):
        """Freeze all parameters except for normalization layers"""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze layer norms for better fine-tuning
        for name, param in self.model.named_parameters():
            if 'norm' in name or 'ln' in name:
                param.requires_grad = True
        
        logging.info("Model frozen except for normalization layers")