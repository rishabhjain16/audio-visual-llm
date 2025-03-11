import os
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from datetime import datetime
import json
import traceback
from contextlib import nullcontext
from pathlib import Path
import gc

class ClipWhisperTrainer:
    """
    A trainer for the ClipWhisper model that supports multiple modalities.
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_epochs=10,
        output_dir="outputs/clip_whisper",
        device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=False,
        grad_accum_steps=4,
        log_interval=10,
        save_every=1,
        save_steps=None,
        max_grad_norm=0.5,
        warmup_steps=0,
        log_param_updates=False,
    ):
        """Initialize the trainer with a model, data, and training parameters"""
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.output_dir = output_dir
        self.device = device
        self.fp16 = fp16 and torch.cuda.is_available()
        self.grad_accum_steps = grad_accum_steps
        self.log_interval = log_interval
        self.save_every = save_every
        self.save_steps = save_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.log_param_updates = log_param_updates
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
        # Set up optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        
        # Initialize training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Log configuration
        self._log_training_info()
        
        # Keep track of losses
        self.train_losses = []
        self.val_losses = []
        
        # Create logs directory
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        # Log training parameters
        try:
            if hasattr(self.train_dataloader, 'dataset') and hasattr(self.train_dataloader.dataset, '__len__'):
                logging.info(f"Trainer initialized with {self.train_dataloader.dataset.__len__()} training samples")
            else:
                logging.info(f"Trainer initialized with {len(self.train_dataloader)} training batches")
                
            if val_dataloader and hasattr(val_dataloader, 'dataset') and hasattr(val_dataloader.dataset, '__len__'):
                logging.info(f"Validation set has {val_dataloader.dataset.__len__()} samples")
            elif val_dataloader:
                logging.info(f"Validation set has {len(val_dataloader)} batches")
        except Exception as e:
            logging.warning(f"Could not determine dataset size: {e}")
            
        logging.info(f"Using device: {device}, FP16: {fp16}")
        logging.info(f"Training for {max_epochs} epochs with LR: {learning_rate}")
        logging.info(f"Gradient accumulation steps: {grad_accum_steps}, Max grad norm: {max_grad_norm}")
        
    def _setup_optimizer(self):
        """Set up optimizer and scheduler with more stable settings"""
        # Filter parameters that require gradients and group them
        # Group 1: parameters that should have weight decay
        # Group 2: bias terms, LayerNorm, etc. that should not have weight decay
        decay_params = []
        no_decay_params = []
        
        # Log parameter counts and settings
        logging.info("Setting up optimizer with stable settings for training")
        
        # Group parameters based on whether they should have weight decay
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Don't apply weight decay to bias terms, LayerNorm, or embeddings
            if 'bias' in name or 'layer_norm' in name or 'layernorm' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Set up parameter groups with appropriate weight decay
        param_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        # Initialize optimizer with conservative settings for stability
        # - Lower beta2 (0.95 instead of 0.999) for faster adaptation to gradient changes
        # - Higher epsilon (1e-8 instead of 1e-8) for numerical stability
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.95),  # Conservative beta2 for better stability
            eps=1e-8,           # Higher epsilon for numerical stability
        )
        
        # Set up cosine learning rate scheduler with warmup
        if self.warmup_steps > 0:
            # Calculate total steps
            num_batches = len(self.train_dataloader)
            total_steps = self.max_epochs * num_batches
            
            # Create scheduler with warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
            
            logging.info(f"Using cosine scheduler with {self.warmup_steps} warmup steps over {total_steps} total steps")
        else:
            # Simple cosine annealing scheduler
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.max_epochs * len(self.train_dataloader)
            )
            
            logging.info("Using cosine annealing scheduler without warmup")
        
        return self.optimizer
    
    def _setup_scheduler(self):
        """Set up the learning rate scheduler"""
        # Calculate total steps
        num_training_steps = len(self.train_dataloader) * self.max_epochs // self.grad_accum_steps
        
        # Create scheduler with warmup
        num_warmup_steps = int(num_training_steps * 0.1) if self.warmup_steps is None else self.warmup_steps
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logging.info(f"Scheduler: Linear warmup with decay")
        logging.info(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
        
        return scheduler
    
    def train(self):
        """Train the model"""
        # Track loss
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        # Track step-wise losses for more detailed monitoring
        step_train_losses = []
        
        # Set up a log file for tracking losses
        loss_log_path = os.path.join(self.output_dir, "loss_log.txt")
        with open(loss_log_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
        
        # Print header for console logging
        print("\n" + "=" * 80)
        print(f"{'Epoch':^10}|{'Train Loss':^20}|{'Val Loss':^20}|{'Best Val':^20}")
        print("-" * 80)
        
        # Get modality information from model for logging
        modality = getattr(self.model, 'modality', 'unknown')
        logging.info(f"Training with modality: {modality}")
        
        # Display active component summary based on modality
        self._display_active_components_summary(modality)
        
        try:
            for epoch in range(self.max_epochs):
                # Clear cache at start of epoch
                torch.cuda.empty_cache()
                gc.collect()
                
                # Train and validate
                train_loss = self._train_epoch(epoch)
                val_loss = self._validate() if self.val_dataloader else float('inf')
                
                # Clear cache after each epoch
                torch.cuda.empty_cache()
                gc.collect()
                
                # Save checkpoint based on epoch if step-based saving is not enabled
                if (epoch + 1) % self.save_every == 0:
                    self._save_checkpoint()
                    logging.info(f"Checkpoint saved at epoch {epoch+1}")
                
                # Log progress to file
                with open(loss_log_path, "a") as f:
                    f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")
                
                # Log progress to console with clear formatting
                is_best = val_loss == best_val_loss
                best_marker = " (BEST)" if is_best else ""
                print(f"{epoch+1:^10}|{train_loss:^20.6f}|{val_loss:^20.6f}|{best_val_loss:^20.6f}{best_marker}")
                
                # Provide regular logging output
                modality_info = f"[Modality: {modality}] "
                if self.val_dataloader is not None:
                    logging.info(f"{modality_info}Epoch {epoch+1}/{self.max_epochs} - "
                               f"Train loss: {train_loss:.6f}, "
                               f"Val loss: {val_loss:.6f}")
                else:
                    logging.info(f"{modality_info}Epoch {epoch+1}/{self.max_epochs} - "
                               f"Train loss: {train_loss:.6f}")
            
            # Print final summary
            print("=" * 80)
            print(f"Training completed after {self.max_epochs} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Final training loss: {train_losses[-1]:.6f}")
            print("=" * 80)
            
            # Save final model
            self._save_checkpoint(is_final=True)
            
            # Plot loss
            self._plot_loss(train_losses, val_losses)
            
            return {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
            }
        finally:
            # Ensure all memory is cleared
            torch.cuda.empty_cache()
            gc.collect()
    
    def _train_epoch(self, epoch):
        """Train for one epoch with simplified logging"""
        self.model.train()
        train_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        # Initialize gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        
        tqdm_desc = f"Epoch {epoch+1}/{self.max_epochs}"
        pbar = tqdm(enumerate(self.train_dataloader), total=num_batches, desc=tqdm_desc)
        
        for batch_idx, batch in pbar:
            # Process batch
            batch_dict = self._process_batch(batch, batch_idx, is_train=True)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.fp16):
                outputs = self.model(**batch_dict)
                loss = outputs["loss"] / self.grad_accum_steps
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            # Update every grad_accum_steps or at end of epoch
            if (batch_idx + 1) % self.grad_accum_steps == 0 or batch_idx == num_batches - 1:
                scaler.step(self.optimizer)
                scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Track loss
            train_loss += loss.item() * self.grad_accum_steps
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{train_loss / (batch_idx + 1):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Only log every 100 batches
            if batch_idx % 100 == 0:
                logging.debug(f"Batch {batch_idx} - Loss: {loss.item():.4f}")
        
        return train_loss / num_batches
    
    def _validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        # Get modality information for logging
        modality = getattr(self.model, 'modality', 'unknown')
        
        # Progress bar
        pbar = tqdm(
            enumerate(self.val_dataloader),
            total=len(self.val_dataloader),
            desc="Validation"
        )
        
        # Validation loop
        with torch.no_grad():
            for batch_idx, batch in pbar:
                try:
                    # Skip None batches (from collate_fn)
                    if batch is None:
                        continue
                    
                    # Process batch
                    loss = self._process_batch(batch, batch_idx=batch_idx, is_train=False)
                    
                    # Update progress bar
                    pbar.set_description(f"Validation | Loss: {loss:.6f} | Modality: {modality}")
                    
                    # Update statistics
                    batch_size = len(batch["audio"])
                    total_loss += loss * batch_size
                    total_samples += batch_size
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {e}")
                    logging.error(traceback.format_exc())
                    continue
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        # Log validation results
        logging.info(f"Validation complete | Avg Loss: {avg_loss:.6f} | Modality: {modality}")
        
        return avg_loss
    
    def _process_batch(self, batch, batch_idx=0, is_train=True):
        """Process a batch of data"""
        # Skip None batches (from collate_fn)
        if batch is None:
            if is_train:
                return {"loss": torch.tensor(0.0, device=self.device)}
            else:
                return {"loss": torch.tensor(0.0, device=self.device)}
        
        # Move batch to device - handle both tuple and dict batch formats
        if isinstance(batch, (list, tuple)) and len(batch) >= 4:
            audio, video, text, prompt = batch[:4]
            
            # Convert text to tensor if it's a list
            if isinstance(text, list):
                # Tokenize text if we have a tokenizer
                if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                    text = self.model.tokenizer(
                        text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=self.model.max_seq_len
                    ).input_ids
                else:
                    # Convert list of strings to tensor of token IDs
                    text = torch.tensor(text, dtype=torch.long)
            
            # Create batch dictionary for the model
            batch_dict = {
                "audio": audio.to(self.device) if audio is not None else None,
                "video": video.to(self.device) if video is not None else None,
                "labels": text.to(self.device) if text is not None else None,
                "return_loss": True
            }
            
            # Add prompt if available
            if prompt is not None:
                if isinstance(prompt, torch.Tensor):
                    batch_dict["prompt"] = prompt.to(self.device)
                elif isinstance(prompt, list) and all(isinstance(p, torch.Tensor) for p in prompt):
                    batch_dict["prompt"] = torch.stack([p.to(self.device) for p in prompt])
                elif isinstance(prompt, list) and all(isinstance(p, (list, tuple)) for p in prompt):
                    batch_dict["prompt"] = torch.tensor(prompt, device=self.device)
                elif isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
                    if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                        encoded = self.model.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.model.max_seq_len)
                        batch_dict["prompt"] = encoded.input_ids.to(self.device)
            
            # Only check dimensions if the inputs are not None
            if audio is not None and video is not None:
                assert audio.size(0) == video.size(0), "Audio and video batch sizes must match"
            if text is not None:
                if audio is not None:
                    assert audio.size(0) == text.size(0), "Input and label batch sizes must match"
                elif video is not None:
                    assert video.size(0) == text.size(0), "Input and label batch sizes must match"
            
        else:
            # Handle dictionary batch format
            batch_dict = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_dict[k] = v.to(self.device)
                elif k in ["labels", "prompt"] and isinstance(v, list):
                    # Handle list inputs for labels or prompt
                    if all(isinstance(item, torch.Tensor) for item in v):
                        batch_dict[k] = torch.stack([item.to(self.device) for item in v])
                    elif all(isinstance(item, (list, tuple)) for item in v):
                        batch_dict[k] = torch.tensor(v, device=self.device)
                    elif all(isinstance(item, str) for item in v) and hasattr(self.model, 'tokenizer'):
                        encoded = self.model.tokenizer(v, return_tensors="pt", padding=True, truncation=True, max_length=self.model.max_seq_len)
                        batch_dict[k] = encoded.input_ids.to(self.device)
                else:
                    batch_dict[k] = v
            
            # Ensure return_loss is set
            batch_dict["return_loss"] = True
        
        if is_train:
            # In training mode, ensure tokenizer has a pad token
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                if self.model.tokenizer.pad_token is None:
                    self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
                    # Add the pad token to the vocabulary if needed
                    if self.model.tokenizer.pad_token not in self.model.tokenizer.get_vocab():
                        self.model.tokenizer.add_special_tokens({'pad_token': self.model.tokenizer.pad_token})
                    
            # Add gradient checkpointing
            if self.grad_accum_steps > 1:
                torch.utils.checkpoint.checkpoint(self.model, **batch_dict)
            
            return batch_dict
        else:
            # In validation mode, compute loss manually
            try:
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    return outputs  # Return full outputs for validation metrics
            except Exception as e:
                logging.error(f"Error during validation batch {batch_idx}: {e}")
                logging.error(traceback.format_exc())
                return {"loss": torch.tensor(float('inf'), device=self.device)}
    
    def _save_checkpoint(self, is_best=False, is_final=False):
        """Save a checkpoint of the model and training state"""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_step_{self.global_step}")
        if is_final:
            checkpoint_dir = os.path.join(self.output_dir, "final_model")
        elif is_best:
            checkpoint_dir = os.path.join(self.output_dir, "best_model")
            
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(checkpoint_dir, "model")
        os.makedirs(model_path, exist_ok=True)
        self.model.save_pretrained(model_path)
        
        # Save training state (optimizer, scheduler, etc.)
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "fp16": self.fp16
        }
        
        # Save scaler state if using fp16
        if self.fp16 and self.scaler:
            training_state["scaler"] = self.scaler.state_dict()
            
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        logging.info(f"Saved checkpoint to {checkpoint_dir}")
        return checkpoint_dir
    
    def load_checkpoint(self, checkpoint_dir):
        """Load a checkpoint of the model and training state"""
        try:
            # Load model
            model_path = os.path.join(checkpoint_dir, "model")
            if os.path.exists(model_path):
                self.model = self.model.__class__.from_pretrained(model_path)
                self.model.to(self.device)
                logging.info(f"Loaded model from {model_path}")
            else:
                logging.warning(f"No model found at {model_path}")
                
            # Load training state
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location=self.device)
                
                # Load optimizer state
                if "optimizer" in training_state:
                    self.optimizer.load_state_dict(training_state["optimizer"])
                    logging.info("Loaded optimizer state")
                    
                # Load scheduler state
                if "scheduler" in training_state and training_state["scheduler"] and self.scheduler:
                    self.scheduler.load_state_dict(training_state["scheduler"])
                    logging.info("Loaded scheduler state")
                    
                # Load scaler state if using fp16
                if self.fp16 and "scaler" in training_state and self.scaler:
                    self.scaler.load_state_dict(training_state["scaler"])
                    logging.info("Loaded gradient scaler state for mixed precision training")
                    
                # Load other training state variables
                if "global_step" in training_state:
                    self.global_step = training_state["global_step"]
                    logging.info(f"Resuming from global step {self.global_step}")
                    
                if "epoch" in training_state:
                    self.current_epoch = training_state["epoch"]
                    logging.info(f"Resuming from epoch {self.current_epoch}")
                    
                if "best_val_loss" in training_state:
                    self.best_val_loss = training_state["best_val_loss"]
                    logging.info(f"Best validation loss: {self.best_val_loss:.4f}")
                    
                return True
            else:
                logging.warning(f"No training state found at {training_state_path}")
                return False
                
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            logging.error(traceback.format_exc())
            return False
    
    def _plot_loss(self, train_losses, val_losses):
        """Plot loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        
        if val_losses:
            plt.plot(val_losses, label="Validation Loss")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(os.path.join(self.output_dir, f"loss_{timestamp}.png"))
        
        # Save data
        with open(os.path.join(self.output_dir, f"loss_{timestamp}.json"), "w") as f:
            json.dump(
                {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                },
                f
            )
    
    def _display_active_components_summary(self, modality):
        """Display summary of active model components"""
        try:
            # Get dimensions from model
            audio_dim = getattr(self.model, 'audio_dim', None)
            video_dim = getattr(self.model, 'video_dim', None)
            llm_dim = getattr(self.model, 'llm_dim', None)
            
            # Format for display (handle None values)
            audio_dim_str = str(audio_dim) if audio_dim is not None else "N/A"
            video_dim_str = str(video_dim) if video_dim is not None else "N/A"
            llm_dim_str = str(llm_dim) if llm_dim is not None else "N/A"
            
            # Display table of active components
            logging.info("\nACTIVE COMPONENTS FOR MODALITY: {}".format(modality.upper()))
            logging.info(f"{'COMPONENT':<30} {'INPUT':<15} {'OUTPUT':<15} {'ACTIVE':<10}")
            logging.info("-" * 75)
            
            # Display encoders based on modality
            audio_active = modality in ["audio", "both"]
            video_active = modality in ["video", "both"]
            
            # Audio encoder
            if audio_active:
                logging.info(f"{'Audio Encoder (Whisper)':<30} {'Raw Audio':<15} {audio_dim_str:<15} {'YES':<10}")
            else:
                logging.info(f"{'Audio Encoder (Whisper)':<30} {'Raw Audio':<15} {audio_dim_str:<15} {'NO':<10}")
                
            # Video encoder
            if video_active:
                logging.info(f"{'Video Encoder (CLIP)':<30} {'Raw Video':<15} {video_dim_str:<15} {'YES':<10}")
            else:
                logging.info(f"{'Video Encoder (CLIP)':<30} {'Raw Video':<15} {video_dim_str:<15} {'NO':<10}")
            
            # Display connections between components
            if audio_active:
                logging.info(f"{'Audio Connector':<30} {audio_dim_str:<15} {llm_dim_str:<15} {'YES':<10}")
            else:
                logging.info(f"{'Audio Connector':<30} {audio_dim_str:<15} {llm_dim_str:<15} {'NO':<10}")
                
            if video_active:
                logging.info(f"{'Video Connector':<30} {video_dim_str:<15} {llm_dim_str:<15} {'YES':<10}")
            else:
                logging.info(f"{'Video Connector':<30} {video_dim_str:<15} {llm_dim_str:<15} {'NO':<10}")
            
            # LLM is always active
            logging.info(f"{'Language Model (LLM)':<30} {llm_dim_str:<15} {'Text':<15} {'YES':<10}")
            
            # Memory usage summary
            logging.info("\nESTIMATED MEMORY USAGE:")
            
            # These are rough estimates based on model sizes
            whisper_mem = "~0.5 GB" if audio_active else "0 GB (not loaded)"
            clip_mem = "~0.5 GB" if video_active else "0 GB (not loaded)"
            llm_mem = "~3.5 GB"  # 1B model with activations
            batch_mem = "~0.1 GB"  # Memory for inputs, gradients, etc.
            
            logging.info(f"• Whisper: {whisper_mem}")
            logging.info(f"• CLIP: {clip_mem}")
            logging.info(f"• LLM: {llm_mem}")
            logging.info(f"• Batch data and activations: {batch_mem}")
            
            total_est = ("~5 GB" if modality == "both" else 
                         "~4 GB" if modality == "audio" else 
                         "~4 GB" if modality == "video" else "Unknown")
            
            logging.info(f"• Total estimated usage: {total_est}")
            logging.info("-" * 75)
            
        except Exception as e:
            logging.error(f"Error displaying component summary: {e}")
    
    def _log_training_info(self):
        """Log essential training configuration"""
        logging.info("=" * 50)
        logging.info("Training Configuration:")
        logging.info(f"• Epochs: {self.max_epochs}")
        logging.info(f"• Learning Rate: {self.learning_rate}")
        logging.info(f"• Batch Size: {self.train_dataloader.batch_size}")
        logging.info(f"• Device: {self.device}")
        logging.info(f"• FP16: {self.fp16}")
        logging.info("=" * 50)

    def _log_memory_usage(self, stage):
        # Add detailed memory tracking
        memory = torch.cuda.memory_stats()
        logging.info(f"Memory Details [{stage}]:")
        logging.info(f"- Allocated: {memory['allocated_bytes.all.current'] / 1e9:.2f} GB")
        logging.info(f"- Reserved: {memory['reserved_bytes.all.current'] / 1e9:.2f} GB")
        logging.info(f"- Active: {memory['active_bytes.all.current'] / 1e9:.2f} GB")
        logging.info(f"- Peak: {memory['allocated_bytes.all.peak'] / 1e9:.2f} GB")

    def _find_optimal_batch_size(self):
        """Find the optimal batch size using actual data from the dataloader"""
        if not self.train_dataloader:
            return 1  # Default to batch size 1 if no dataloader
        
        # Get the first batch from the dataloader
        try:
            first_batch = next(iter(self.train_dataloader))
        except Exception as e:
            logging.warning(f"Could not get first batch from dataloader: {e}")
            return 1
        
        batch_size = 1
        while True:
            try:
                # Create a batch by repeating the first batch
                batch = self._create_batch_from_sample(first_batch, batch_size)
                
                # Test memory usage
                with torch.no_grad():
                    self.model(**batch)
                
                # Double the batch size
                batch_size *= 2
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    return max(1, batch_size // 2)
                raise

    def _create_batch_from_sample(self, sample, batch_size):
        """Create a batch by repeating a single sample"""
        if isinstance(sample, (list, tuple)):
            return [self._repeat_tensor(x, batch_size) if torch.is_tensor(x) else x for x in sample]
        elif isinstance(sample, dict):
            return {k: self._repeat_tensor(v, batch_size) if torch.is_tensor(v) else v for k, v in sample.items()}
        else:
            return sample

    def _repeat_tensor(self, tensor, batch_size):
        """Repeat a tensor along the batch dimension"""
        if tensor.dim() == 0:
            return tensor.unsqueeze(0).repeat(batch_size)
        return tensor.unsqueeze(0).repeat(batch_size, *[1]*(tensor.dim()-1)) 