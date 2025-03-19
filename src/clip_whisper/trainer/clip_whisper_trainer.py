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
import shutil

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
        grad_clip=0.5,
        warmup_steps=0,
        log_param_updates=False,
    ):
        """Initialize the trainer with a model, data, and training parameters"""
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Ensure learning_rate is a float
        try:
            self.learning_rate = float(learning_rate)
        except (ValueError, TypeError):
            logging.warning(f"Could not convert learning rate '{learning_rate}' to float, using default 1e-5")
            self.learning_rate = 1e-5
        
        # Ensure weight_decay is a float
        try:
            self.weight_decay = float(weight_decay)
        except (ValueError, TypeError):
            logging.warning(f"Could not convert weight_decay '{weight_decay}' to float, using default 0.01")
            self.weight_decay = 0.01
        
        # Ensure max_epochs is an int
        try:
            self.max_epochs = int(max_epochs)
        except (ValueError, TypeError):
            logging.warning(f"Could not convert max_epochs '{max_epochs}' to int, using default 10")
            self.max_epochs = 10
        
        self.output_dir = output_dir
        self.device = device
        self.fp16 = fp16 and torch.cuda.is_available()
        
        # Ensure grad_accum_steps is an int
        try:
            self.grad_accum_steps = int(grad_accum_steps)
        except (ValueError, TypeError):
            logging.warning(f"Could not convert grad_accum_steps '{grad_accum_steps}' to int, using default 4")
            self.grad_accum_steps = 4
        
        # Ensure log_interval is an int
        try:
            self.log_interval = int(log_interval) if log_interval is not None else 10
        except (ValueError, TypeError):
            logging.warning(f"Could not convert log_interval '{log_interval}' to int, using default 10")
            self.log_interval = 10
        
        # Ensure save_every is an int
        try:
            self.save_every = int(save_every) if save_every is not None else 1
        except (ValueError, TypeError):
            logging.warning(f"Could not convert save_every '{save_every}' to int, using default 1")
            self.save_every = 1
        
        # Ensure save_steps is an int or None
        if save_steps is not None:
            try:
                self.save_steps = int(save_steps)
            except (ValueError, TypeError):
                logging.warning(f"Could not convert save_steps '{save_steps}' to int, using None")
                self.save_steps = None
        else:
            self.save_steps = None
        
        # Ensure grad_clip is a float
        try:
            self.grad_clip = float(grad_clip)
        except (ValueError, TypeError):
            logging.warning(f"Could not convert grad_clip '{grad_clip}' to float, using default 0.5")
            self.grad_clip = 0.5
        
        # Ensure warmup_steps is an int
        try:
            self.warmup_steps = int(warmup_steps)
        except (ValueError, TypeError):
            logging.warning(f"Could not convert warmup_steps '{warmup_steps}' to int, using default 0")
            self.warmup_steps = 0
        
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
        logging.info(f"Gradient accumulation steps: {grad_accum_steps}, Max grad clip: {grad_clip}")
        
        # Add this line
        self.global_step = 0
        
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
        """Train the model with time-based checkpoints and adaptive monitoring"""
        # Initialize checkpoint timing and monitoring
        last_checkpoint_time = time.time()
        checkpoint_interval_hours = 2  # Save checkpoint every 2 hours
        train_start_time = time.time()
        
        # Create loss tracking for stability monitoring
        loss_history = []
        unstable_count = 0
        
        # Loss tracking per epoch
        self.train_losses = []
        self.val_losses = []
        
        # Save path for loss log
        loss_log_path = os.path.join(self.output_dir, "loss_log.csv")
        
        # Write header to loss log if it doesn't exist
        if not os.path.exists(loss_log_path):
            with open(loss_log_path, "w") as f:
                f.write("epoch,train_loss,val_loss,time_hours,remaining_hours\n")
        
        # Display training header
        print(f"{'Epoch':<8}|{'Train Loss':<15}|{'Val Loss':<15}|{'Best Val':<15}|{'Time (h)':<10}|{'ETA (h)':<10}")
        print("-" * 80)
        
        # Initialize best validation loss
        best_val_loss = float('inf')
        
        try:
            for epoch in range(self.max_epochs):
                self.current_epoch = epoch
                
                # Clear cache at start of epoch
                # torch.cuda.empty_cache()
                # gc.collect()
                
                # Train and validate
                epoch_start = time.time()
                train_loss = self._train_epoch(epoch)
                self.train_losses.append(train_loss)
                
                if self.val_dataloader:
                    val_loss = self._validate()
                    self.val_losses.append(val_loss)
                    
                    # Update best validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # Save best model
                        self._save_checkpoint(is_best=True)
                        logging.info(f"New best validation loss: {best_val_loss:.6f}")
                else:
                    val_loss = float('inf')
                    self.val_losses.append(val_loss)
                
                # Time-based checkpoint
                current_time = time.time()
                hours_since_checkpoint = (current_time - last_checkpoint_time) / 3600
                
                # Save checkpoint based on time or epoch schedule
                if hours_since_checkpoint >= checkpoint_interval_hours:
                    checkpoint_path = self._save_checkpoint()
                    logging.info(f"Time-based checkpoint saved after {hours_since_checkpoint:.2f} hours to {checkpoint_path}")
                    last_checkpoint_time = current_time
                elif (epoch + 1) % self.save_every == 0:
                    checkpoint_path = self._save_checkpoint()
                    logging.info(f"Epoch-based checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
                    last_checkpoint_time = current_time
                
                # Calculate training statistics and ETA
                epoch_time = time.time() - epoch_start
                total_time = time.time() - train_start_time
                total_hours = total_time / 3600
                
                # Calculate remaining time
                progress = (epoch + 1) / self.max_epochs
                if progress > 0:
                    estimated_total = total_time / progress
                    remaining_time = estimated_total - total_time
                    remaining_hours = remaining_time / 3600
                else:
                    remaining_hours = 0
                
                # Monitor loss stability
                loss_history.append(train_loss)
                if len(loss_history) > 5:
                    loss_history.pop(0)
                    
                    # Check if loss is stable
                    if not np.isfinite(train_loss) or train_loss > 1e6:
                        unstable_count += 1
                        if unstable_count >= 3:
                            logging.error("Unstable training detected. Loss is too high or not finite.")
                            self._save_checkpoint(is_final=False)
                            logging.info("Emergency checkpoint saved. Training may need to be stopped.")
                    else:
                        unstable_count = 0
                
                # Log progress to file
                with open(loss_log_path, "a") as f:
                    f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{total_hours:.2f},{remaining_hours:.2f}\n")
                
                # Log progress to console with clear formatting
                is_best = val_loss == best_val_loss
                best_marker = "*" if is_best else ""
                print(f"{epoch+1:<8}|{train_loss:<15.6f}|{val_loss:<15.6f}|{best_val_loss:<15.6f}{best_marker}|{total_hours:<10.2f}|{remaining_hours:<10.2f}")
                
                # Detailed logging
                modality_info = f"[Modality: {getattr(self.model, 'modality', 'unknown')}]"
                logging.info(f"{modality_info} Epoch {epoch+1}/{self.max_epochs} - "
                           f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
                           f"Time: {total_hours:.2f}h, ETA: {remaining_hours:.2f}h")
                
                # Clear cache after each epoch
                torch.cuda.empty_cache()
                gc.collect()
            
            # Save final model
            self._save_checkpoint(is_final=True)
            
            # Print final summary
            total_time = time.time() - train_start_time
            total_hours = total_time / 3600
            print("\n" + "=" * 80)
            print(f"Training completed after {self.max_epochs} epochs ({total_hours:.2f} hours)")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Final training loss: {self.train_losses[-1]:.6f}")
            print("=" * 80)
            
            # Plot loss curves
            self._plot_loss(self.train_losses, self.val_losses)
            
            return {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": best_val_loss,
                "training_hours": total_hours
            }
        
        except Exception as e:
            # Save emergency checkpoint
            logging.error(f"Training interrupted by error: {e}")
            logging.error(traceback.format_exc())
            try:
                emergency_dir = os.path.join(self.output_dir, "emergency_checkpoint")
                self._save_checkpoint(checkpoint_dir=emergency_dir)
                logging.info(f"Emergency checkpoint saved to {emergency_dir}")
            except Exception as save_error:
                logging.error(f"Could not save emergency checkpoint: {save_error}")
            
            raise
        
        # finally:
        #     # Ensure all memory is cleared
        #     torch.cuda.empty_cache()
        #     gc.collect()
    
    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_batches = 0
        nan_found = False
        unstable_batch_count = 0
        
        # Reserve memory at the start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            reserve_size = 1024 * 1024 * 1024  # 1GB
            self._memory_reserve = torch.empty(reserve_size, dtype=torch.uint8, device='cuda')
        
        # Create progress bar
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch {epoch+1}/{self.max_epochs}"
        )
        
        for batch_idx, batch in pbar:
            try:
                # Process batch
                outputs = self._process_batch(batch, batch_idx)
                
                # Extract loss from outputs dictionary
                if isinstance(outputs, dict):
                    loss = outputs["loss"]
                else:
                    loss = outputs
                
                if torch.isnan(loss):
                    logging.warning(f"NaN loss detected in batch {batch_idx}")
                    nan_found = True
                    unstable_batch_count += 1
                    if unstable_batch_count > 5:
                        logging.error("Too many unstable batches. Stopping epoch.")
                        break
                    continue
                
                # Backward pass and optimization
                loss.backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Step optimizer and scheduler
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
                
                # Update totals
                total_loss += loss.item()
                total_batches += 1
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_loss = total_loss / total_batches
                progress = (batch_idx + 1) / len(self.train_dataloader) * 100
                
                # Update progress bar description
                pbar.set_description(
                    f"Epoch {epoch+1}/{self.max_epochs} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Avg: {avg_loss:.6f} | "
                    f"LR: {current_lr:.6f}"
                )
                
                # Periodically log memory usage (less frequently)
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    logging.info(f"Memory [Batch {batch_idx}] {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
                
                # Reset unstable counter if we had a successful batch
                unstable_batch_count = 0
                    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logging.error(f"CUDA OOM in batch {batch_idx}. Trying to recover...")
                    if hasattr(self, '_memory_reserve'):
                        del self._memory_reserve  # Free the reserved memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    logging.error(f"Error in batch {batch_idx}: {e}")
                    logging.error(traceback.format_exc())
                    continue
                    
            except Exception as e:
                logging.error(f"Unexpected error in batch {batch_idx}: {e}")
                logging.error(traceback.format_exc())
                continue
        
        # Clean up memory reservation
        if hasattr(self, '_memory_reserve'):
            del self._memory_reserve
        
        # Calculate average loss
        if total_batches == 0:
            logging.error("No successful batches in this epoch!")
            return float('inf')
        
        if nan_found:
            logging.warning("NaN losses detected during training. Results may be unreliable.")
        
        avg_loss = total_loss / total_batches
        logging.info(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.6f}")
        
        return avg_loss
    
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
                    
                    # Process batch based on format
                    if isinstance(batch, dict):
                        # Get batch size from dict
                        audio_batch_size = batch["audio"].size(0) if "audio" in batch and batch["audio"] is not None else 0
                        video_batch_size = batch["video"].size(0) if "video" in batch and batch["video"] is not None else 0
                        batch_size = max(audio_batch_size, video_batch_size)
                    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        # Get batch size from tuple elements
                        elements = [b for b in batch if isinstance(b, torch.Tensor)]
                        if elements:
                            batch_size = elements[0].size(0)
                        else:
                            batch_size = 1
                    else:
                        # Default batch size
                        batch_size = 1
                    
                    # Process batch using the _process_batch method
                    outputs = self._process_batch(batch, batch_idx=batch_idx, is_train=False)
                    
                    # Get loss value
                    if isinstance(outputs, dict) and "loss" in outputs:
                        loss_value = outputs["loss"]
                    elif isinstance(outputs, torch.Tensor):
                        loss_value = outputs
                    else:
                        logging.warning(f"Unexpected output type from _process_batch: {type(outputs)}")
                        loss_value = torch.tensor(0.0, device=self.device)
                    
                    # Add stability checks
                    if torch.isnan(loss_value) or torch.isinf(loss_value):
                        logging.warning(f"Found unstable loss value: {loss_value}. Setting to large finite value.")
                        loss_value = torch.tensor(1e6, device=loss_value.device, dtype=loss_value.dtype)
                    
                    # Update progress bar
                    pbar.set_description(f"Validation | Loss: {loss_value.item():.6f} | Modality: {modality}")
                    
                    # Update statistics
                    total_loss += loss_value.item() * batch_size
                    total_samples += batch_size
                    
                except Exception as e:
                    logging.error(f"Error in validation batch {batch_idx}: {e}")
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
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Move batch to device - handle both tuple and dict batch formats
        if isinstance(batch, (list, tuple)) and len(batch) >= 4:
            audio, video, text, tokens = batch[:4]
            
            # Use the tokenized version (tokens) as labels, not the raw text
            labels = tokens
            
            # Convert labels to tensor if it's a list
            if isinstance(labels, list):
                # If it's a list of lists (tokenized text), convert to tensor
                if all(isinstance(item, list) for item in labels):
                    try:
                        labels = torch.tensor(labels, dtype=torch.long)
                        logging.debug(f"Converted list of token lists to tensor with shape {labels.shape}")
                    except Exception as e:
                        logging.error(f"Failed to convert token lists to tensor: {e}")
                        # Fall back to tokenizing the raw text
                        if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                            logging.warning("Falling back to tokenizing raw text")
                            labels = self.model.tokenizer(
                                text, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=self.model.max_seq_len
                            ).input_ids
                elif all(isinstance(item, str) for item in labels):
                    # If we somehow got raw text strings, tokenize them
                    if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                        logging.warning("Received raw text strings as labels, tokenizing")
                        labels = self.model.tokenizer(
                            labels, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=self.model.max_seq_len
                        ).input_ids
                    else:
                        logging.error("Received string labels but no tokenizer available")
                        # Create dummy labels to avoid crashing
                        batch_size = len(audio) if audio is not None else (len(video) if video is not None else 1)
                        labels = torch.zeros((batch_size, 10), dtype=torch.long)
            
            # Create batch dictionary for the model
            batch_dict = {
                "audio": audio.to(self.device) if audio is not None else None,
                "video": video.to(self.device) if video is not None else None,
                "labels": labels.to(self.device) if labels is not None else None,
                "return_loss": True
            }
            
            # Add prompt if available
            if text is not None:
                if isinstance(text, torch.Tensor):
                    batch_dict["prompt"] = text.to(self.device)
                elif isinstance(text, list) and all(isinstance(p, torch.Tensor) for p in text):
                    batch_dict["prompt"] = torch.stack([p.to(self.device) for p in text])
                elif isinstance(text, list) and all(isinstance(p, (list, tuple)) for p in text):
                    batch_dict["prompt"] = torch.tensor(text, device=self.device)
                elif isinstance(text, list) and all(isinstance(p, str) for p in text) and hasattr(self.model, 'tokenizer'):
                    encoded = self.model.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.model.max_seq_len)
                    batch_dict["prompt"] = encoded.input_ids.to(self.device)
            
            # Only check dimensions if the inputs are not None
            if audio is not None and video is not None:
                assert audio.size(0) == video.size(0), "Audio and video batch sizes must match"
            if labels is not None:
                if audio is not None:
                    assert audio.size(0) == labels.size(0), "Input and label batch sizes must match"
                elif video is not None:
                    assert video.size(0) == labels.size(0), "Input and label batch sizes must match"
            
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
            # Forward pass through the model
            outputs = self.model(**batch_dict)
            
            # Extract loss from outputs
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs
                
            return loss
        else:
            # In validation mode, compute loss manually
            try:
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    return outputs  # Return full outputs for validation metrics
            except Exception as e:
                logging.error(f"Error during validation batch {batch_idx}: {e}")
                logging.error(traceback.format_exc())
                return torch.tensor(float('inf'), device=self.device)
    
    def _save_checkpoint(self, is_final=False, is_best=False, checkpoint_dir=None):
        """Save a checkpoint of the model and optimizer state.
        
        Args:
            is_final (bool): Whether this is the final checkpoint after training
            is_best (bool): Whether this checkpoint has the best validation loss
            checkpoint_dir (str): Optional directory to save the checkpoint in
        
        Returns:
            str: Path to the saved checkpoint
        """
        # Use specified checkpoint dir or default
        checkpoint_dir = checkpoint_dir or self.output_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Determine checkpoint filename based on type
        if is_final:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_final.pt")
            checkpoint_meta_path = os.path.join(checkpoint_dir, f"model_final_meta.json")
        elif is_best:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_best.pt")
            checkpoint_meta_path = os.path.join(checkpoint_dir, f"model_best_meta.json")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{self.current_epoch + 1}.pt")
            checkpoint_meta_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{self.current_epoch + 1}_meta.json")
        
        # Create a unified checkpoint dictionary
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': min(self.val_losses) if self.val_losses else float('inf'),
        }
        
        # Save unified checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata separately for easy inspection without loading the model
        metadata = {
            'epoch': self.current_epoch + 1,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
            'model_config': {
                'modality': getattr(self.model, 'modality', None),
                'model_type': type(self.model).__name__,
            },
            'checkpoint_type': 'final' if is_final else ('best' if is_best else 'regular'),
        }
        
        # Save metadata
        with open(checkpoint_meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # If this is the best model, also save it as model_best
        if is_best and not checkpoint_path.endswith('model_best.pt'):
            best_path = os.path.join(checkpoint_dir, "model_best.pt")
            best_meta_path = os.path.join(checkpoint_dir, "model_best_meta.json")
            shutil.copy(checkpoint_path, best_path)
            shutil.copy(checkpoint_meta_path, best_meta_path)
            logging.info(f"Best model saved to {best_path}")
        
        logging.info(f"Checkpoint saved to {checkpoint_path}")
        
        # For convenience in scripts that process the checkpoint
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint from the specified path.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            dict: Information about the loaded checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logging.info(f"Model state loaded successfully")
        else:
            logging.warning(f"No model state found in checkpoint")
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logging.info(f"Optimizer state loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load optimizer state: {e}")
        
        # Load scheduler state
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logging.info(f"Scheduler state loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load scheduler state: {e}")
        
        # Load training history
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        if "val_losses" in checkpoint:
            self.val_losses = checkpoint["val_losses"]
        
        # Set epoch and step counters
        self.current_epoch = checkpoint.get("epoch", 0)
        
        # Log checkpoint information
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logging.info(f"Resuming from epoch {self.current_epoch}")
        logging.info(f"Best validation loss: {best_val_loss:.6f}")
        
        return {
            "epoch": self.current_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss
        }
    
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