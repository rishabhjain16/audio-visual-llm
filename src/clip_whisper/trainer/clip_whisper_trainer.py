import os
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from datetime import datetime
import json
import traceback
from contextlib import nullcontext
from pathlib import Path

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
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader (optional)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            max_epochs: Maximum number of epochs to train
            output_dir: Directory to save outputs
            device: Device to train on ("cuda" or "cpu")
            fp16: Whether to use mixed precision training
            grad_accum_steps: Gradient accumulation steps
            log_interval: Logging interval (in steps)
            save_every: Save checkpoint every N epochs
            save_steps: Save checkpoint every N steps (overrides save_every)
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps for LR scheduler
            log_param_updates: Whether to log parameter update statistics
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.output_dir = output_dir
        self.device = device
        self.fp16 = fp16
        self.grad_accum_steps = grad_accum_steps
        self.log_interval = log_interval
        self.save_every = save_every
        self.save_steps = save_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.log_param_updates = log_param_updates
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup model, optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup gradient scaler for mixed precision training
        if self.fp16:
            # Use torch.amp.GradScaler instead of torch.cuda.amp.GradScaler
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Initialize step counter
        self.global_step = 0
        
        # Initialize validation loss tracking
        self.best_val_loss = float('inf')
        
        # Setup logging
        self._log_training_info()
        
        # Keep track of losses
        self.train_losses = []
        self.val_losses = []
        
        # Create logs directory
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        # Log training parameters
        logging.info(f"Trainer initialized with {train_dataloader.dataset.__len__()} training samples")
        if val_dataloader:
            logging.info(f"Validation set has {val_dataloader.dataset.__len__()} samples")
        logging.info(f"Using device: {device}, FP16: {fp16}")
        logging.info(f"Training for {max_epochs} epochs with LR: {learning_rate}")
        logging.info(f"Gradient accumulation steps: {grad_accum_steps}, Max grad norm: {max_grad_norm}")
    
    def _setup_optimizer(self):
        """Set up the optimizer for training"""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        # Log which parts of the model have trainable parameters
        no_grad_count = 0
        trainable_count = 0
        
        for name, param in self.model.named_parameters():
            # Skip frozen parameters
            if not param.requires_grad:
                no_grad_count += param.numel()
                continue
                
            # Don't apply weight decay to bias, LayerNorm, and embeddings
            if any(nd in name for nd in ['bias', 'LayerNorm', 'layer_norm', 'embedding']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
            trainable_count += param.numel()
        
        logging.info(f"Total parameters: {no_grad_count + trainable_count:,}")
        logging.info(f"Trainable parameters: {trainable_count:,} ({100 * trainable_count / (no_grad_count + trainable_count):.2f}%)")
        
        # Create parameter groups
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        # Ensure learning_rate is a float
        try:
            # Use a smaller initial learning rate (10% of base rate)
            if isinstance(self.learning_rate, str):
                # Convert string to float if needed
                initial_lr = float(self.learning_rate) * 0.1
            else:
                initial_lr = float(self.learning_rate) * 0.1
            
            logging.info(f"Setting initial learning rate to {initial_lr:.2e}")
        except (ValueError, TypeError) as e:
            # Fallback to default if conversion fails
            logging.warning(f"Error converting learning rate: {e}. Using default value.")
            initial_lr = 5e-6  # Default fallback value
        
        # Create optimizer
        optimizer = AdamW(
            optim_groups,
            lr=initial_lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        logging.info(f"Optimizer: AdamW with initial LR: {initial_lr:.2e}")
        
        return optimizer
    
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
        
        # Training loop
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            logging.info(f"Starting epoch {epoch+1}/{self.max_epochs}")
            
            # Train
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = float('inf')
            if self.val_dataloader is not None:
                val_loss = self._validate()
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(is_best=True)
                    logging.info(f"New best validation loss: {val_loss:.4f}")
            
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
    
    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        # Check if LLM is frozen
        llm_frozen = False
        if hasattr(self.model, 'freeze_llm') and self.model.freeze_llm:
            llm_frozen = True
            logging.info("Training with frozen LLM - training only the connectors")
        
        # Get modality information for logging
        modality = getattr(self.model, 'modality', 'unknown')
        
        # Add periodic sanity check
        logging.info(f"SANITY CHECK - EPOCH {epoch+1}:")
        logging.info(f"  Current modality: {modality}")
        if hasattr(self.model, 'audio_dim'):
            logging.info(f"  Audio dimension: {self.model.audio_dim}")
        if hasattr(self.model, 'video_dim'):
            logging.info(f"  Video dimension: {self.model.video_dim}")
        if hasattr(self.model, 'llm_dim'):
            logging.info(f"  LLM dimension: {self.model.llm_dim}")
        
        # Force print to console as well
        print(f"\nSANITY CHECK - EPOCH {epoch+1}:")
        print(f"  Current modality: {modality}")
        print(f"  Using {modality.upper()} modality for training\n")
        
        # Progress bar
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch {epoch+1}"
        )
        
        # Training loop
        for batch_idx, batch in pbar:
            try:
                # Skip None batches (from collate_fn)
                if batch is None:
                    continue
                
                # Process batch
                loss = self._process_batch(batch, batch_idx=batch_idx, is_train=True)
                
                # Update progress bar
                desc = f"Epoch {epoch+1} | Loss: {loss:.6f}"
                if llm_frozen:
                    desc += " (LLM frozen)"
                desc += f" | Modality: {modality}"
                pbar.set_description(desc)
                
                # Update statistics
                batch_size = len(batch["audio"])
                total_loss += loss * batch_size
                total_samples += batch_size
                
                # Log progress
                if batch_idx % self.log_interval == 0:
                    logging.info(f"Epoch {epoch+1} | Batch {batch_idx}/{len(self.train_dataloader)} | Loss: {loss:.6f} | Modality: {modality}")
                
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                logging.error(traceback.format_exc())
                continue
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
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
        """Process a single batch of data"""
        try:
            # Extract batch data
            audio = batch.get("audio", None)
            video = batch.get("video", None) 
            labels = batch.get("labels", None)
            prompt = batch.get("prompt", None)
            
            # Move batch data to device if needed
            if audio is not None:
                audio = audio.to(self.device)
            if video is not None:
                video = video.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)
            if prompt is not None:
                prompt = prompt.to(self.device)
                
            # Log batch sizes occasionally to track memory usage
            if batch_idx % 50 == 0:
                if audio is not None:
                    logging.info(f"[Batch {batch_idx}] Audio input shape: {audio.shape}")
                if video is not None:
                    logging.info(f"[Batch {batch_idx}] Video input shape: {video.shape}")
                if labels is not None:
                    logging.info(f"[Batch {batch_idx}] Labels shape: {labels.shape}")
            
            # Forward pass through the model
            if is_train:
                # Zero gradients for first accumulation step
                if batch_idx % self.grad_accum_steps == 0:
                    self.optimizer.zero_grad()
                
                # Forward pass with gradient computation
                with torch.amp.autocast('cuda') if self.fp16 else nullcontext():
                    outputs = self.model(
                        audio=audio,
                        video=video,
                        labels=labels,
                        prompt=prompt,
                        return_loss=True,
                        batch_idx=batch_idx,  # Pass batch index for logging
                    )
            else:
                # Validation pass without gradients
                with torch.no_grad():
                    with torch.amp.autocast('cuda') if self.fp16 else nullcontext():
                        outputs = self.model(
                            audio=audio,
                            video=video,
                            labels=labels,
                            prompt=prompt,
                            return_loss=True,
                            batch_idx=batch_idx  # Pass batch index for logging
                        )
            
            # Extract and process loss
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            elif isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                logging.info("No loss found in model outputs (expected when LLM is frozen)")
                # Return a dummy loss of 0.0 that's a proper tensor for stability
                return torch.tensor(0.0, device=self.device).item()
                
            # Additional check to handle None loss (when LLM is frozen)
            if loss is None:
                logging.info("Loss is None (expected when LLM is frozen)")
                return torch.tensor(0.0, device=self.device).item()
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.error(f"NaN/Inf detected in loss at batch {batch_idx}")
                if is_train:
                    # During training, try to recover
                    self.optimizer.zero_grad()  # Clear any bad gradients
                    loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                else:
                    # During validation, skip this batch
                    return 0.0
                    
            # Scale loss for gradient accumulation
            if is_train and self.grad_accum_steps > 1:
                loss = loss / self.grad_accum_steps
                
            # Backward pass with gradient handling
            if is_train:
                try:
                    # Compute gradients
                    if self.fp16:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                    # Check for NaN/Inf in gradients
                    has_bad_gradients = False
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                logging.error(f"NaN/Inf gradients in {name}")
                                
                                # Special handling for LoRA parameters
                                if "lora" in name.lower():
                                    logging.warning(f"Zeroing out bad LoRA gradients in {name}")
                                    param.grad.data.zero_()
                                else:
                                    # For non-LoRA parameters with bad gradients, skip the update
                                    has_bad_gradients = True
                                    break
                                    
                    if has_bad_gradients:
                        logging.warning("Found bad gradients in non-LoRA parameters, skipping update")
                        self.optimizer.zero_grad()
                        return 1.0
                        
                    # Step optimizer and scheduler if needed
                    if (batch_idx + 1) % self.grad_accum_steps == 0 or \
                       (batch_idx + 1) == len(self.train_dataloader):
                        
                        if self.fp16:
                            # First unscale gradients, then clip them, then step optimizer
                            self.scaler.unscale_(self.optimizer)
                            
                            # Clip gradients
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_norm=self.max_grad_norm
                            )
                            
                            # Check for NaN in gradient norm
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                logging.error(f"NaN/Inf gradient norm at batch {batch_idx}")
                                self.optimizer.zero_grad()
                                return 1.0
                                
                            # Step optimizer with scaler
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # For non-FP16 mode, clip gradients and step normally
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_norm=self.max_grad_norm
                            )
                            
                            # Check for NaN in gradient norm
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                logging.error(f"NaN/Inf gradient norm at batch {batch_idx}")
                                self.optimizer.zero_grad()
                                return 1.0
                                
                            # Step optimizer normally
                            self.optimizer.step()
                        
                        # Step scheduler
                        self.scheduler.step()
                        
                        # Log gradient norm occasionally
                        if batch_idx % 100 == 0:
                            logging.info(f"Batch {batch_idx} gradient norm: {grad_norm:.4f}")
                        
                except RuntimeError as e:
                    logging.error(f"Runtime error in backward pass: {e}")
                    self.optimizer.zero_grad()
                    if self.fp16:
                        # Reset scaler state if there was an error
                        self.scaler = torch.amp.GradScaler('cuda')
                    return 1.0
                    
            # Return loss value
            loss_value = loss.item()
            if np.isnan(loss_value) or np.isinf(loss_value):
                logging.warning(f"NaN/Inf in final loss value: {loss_value}")
                return 1.0
                
            # Log additional sequence stats occasionally
            if batch_idx % 100 == 0:
                stats = getattr(self.model, '_seq_len_stats', None)
                if stats:
                    logging.info(f"[Batch {batch_idx}] Sequence length stats - Min: {stats.get('min', 'N/A')}, "
                               f"Max: {stats.get('max', 'N/A')}, Avg: {stats.get('avg', 'N/A')}")
                               
            return loss_value
            
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            logging.error(traceback.format_exc())
            return 1.0  # Default loss on error
    
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
        """Display a summary of active components based on modality"""
        # Make sure we're using the actual model's modality, not the parameter
        if hasattr(self.model, 'modality'):
            modality = self.model.modality
            logging.info(f"Using model's actual modality: {modality}")
        
        # Skip if model doesn't have the expected attributes
        if not hasattr(self.model, 'audio_dim') or not hasattr(self.model, 'video_dim') or not hasattr(self.model, 'llm_dim'):
            return
            
        # Get dimensions
        audio_dim = self.model.audio_dim
        video_dim = self.model.video_dim
        llm_dim = self.model.llm_dim
        
        # Create a clear summary banner
        summary_banner = "=" * 100
        logging.info(f"\n{summary_banner}")
        logging.info(f"ACTIVE COMPONENTS SUMMARY FOR MODALITY: {modality.upper()}")
        logging.info(f"{'-' * 100}")
        
        # Component table header
        logging.info(f"{'COMPONENT':<30} {'INPUT DIM':<15} {'OUTPUT DIM':<15} {'ACTIVE':<10}")
        logging.info(f"{'-' * 100}")
        
        # List components based on modality
        if modality in ["audio", "both"]:
            logging.info(f"{'Audio Encoder (Whisper)':<30} {'Raw Audio':<15} {audio_dim:<15} {'YES':<10}")
            logging.info(f"{'Audio Connector':<30} {audio_dim:<15} {llm_dim:<15} {'YES':<10}")
        else:
            logging.info(f"{'Audio Encoder (Whisper)':<30} {'Raw Audio':<15} {audio_dim:<15} {'NO':<10}")
            logging.info(f"{'Audio Connector':<30} {audio_dim:<15} {llm_dim:<15} {'NO':<10}")
            
        if modality in ["video", "both"]:
            logging.info(f"{'Video Encoder (CLIP)':<30} {'Raw Video':<15} {video_dim:<15} {'YES':<10}")
            logging.info(f"{'Video Connector':<30} {video_dim:<15} {llm_dim:<15} {'YES':<10}")
        else:
            logging.info(f"{'Video Encoder (CLIP)':<30} {'Raw Video':<15} {video_dim:<15} {'NO':<10}")
            logging.info(f"{'Video Connector':<30} {video_dim:<15} {llm_dim:<15} {'NO':<10}")
            
        if modality == "both":
            logging.info(f"{'Fusion (weighted average)':<30} {llm_dim:<15} {llm_dim:<15} {'YES':<10}")
        else:
            logging.info(f"{'Fusion (weighted average)':<30} {llm_dim:<15} {llm_dim:<15} {'NO':<10}")
            
        logging.info(f"{'LLM':<30} {llm_dim:<15} {'Vocab Size':<15} {'YES':<10}")
        
        # End summary
        logging.info(f"{summary_banner}\n")
        
        # Print to console as well for visibility
        print(f"\nACTIVE COMPONENTS FOR MODALITY: {modality.upper()}")
        if modality == "audio":
            print("- Using WHISPER for audio encoding")
            print("- Using audio connector to project to LLM dimension")
        elif modality == "video":
            print("- Using CLIP for video encoding")
            print("- Using video connector to project to LLM dimension")
        elif modality == "both":
            print("- Using WHISPER for audio encoding")
            print("- Using CLIP for video encoding")
            print("- Using both connectors and fusion for combined features")
        print(f"- All features projected to LLM dimension: {llm_dim}")
        print("")

    def _setup_dataloaders(self):
        """Setup data loaders for training and validation"""
        logging.info("Setting up dataloaders")
        
        try:
            # If dataloaders are already provided, use them
            if hasattr(self, 'train_dataloader') and self.train_dataloader is not None:
                logging.info(f"Using provided train dataloader")
                return
            
            # Import dataset modules
            from ..data.dataset import AVSRDataset, create_dataloader
            from ..data.simple_dataset import AVSRDataset as SimpleAVSRDataset, create_dataloaders as create_simple_dataloaders
            
            # Get configuration for data
            if hasattr(self.config, 'data'):
                data_config = self.config.data
                
                # Get data paths
                data_path = getattr(data_config, "path", "data")
                
                # Setup appropriate datasets based on configuration
                # Implementation will depend on your specific requirements
                
                logging.info(f"Dataloaders setup complete")
            else:
                logging.warning("No data configuration found in config")
                
        except Exception as e:
            logging.error(f"Error during dataloader setup: {e}")
            logging.error(traceback.format_exc())

    def _log_training_info(self):
        """Log information about the training configuration"""
        logging.info("=" * 50)
        logging.info("ClipWhisperTrainer Configuration:")
        logging.info("-" * 50)
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Max epochs: {self.max_epochs}")
        logging.info(f"Learning rate: {self.learning_rate}")
        logging.info(f"Weight decay: {self.weight_decay}")
        logging.info(f"Device: {self.device}")
        logging.info(f"FP16 mixed precision: {self.fp16}")
        logging.info(f"Gradient accumulation steps: {self.grad_accum_steps}")
        logging.info(f"Log interval: {self.log_interval}")
        logging.info(f"Save checkpoint every: {self.save_every} epochs")
        if self.save_steps:
            logging.info(f"Save checkpoint every: {self.save_steps} steps")
        logging.info(f"Max gradient norm: {self.max_grad_norm}")
        logging.info(f"Logging parameter updates: {self.log_param_updates}")
        logging.info(f"Training samples: {len(self.train_dataloader.dataset) if hasattr(self.train_dataloader, 'dataset') else 'Unknown'}")
        logging.info(f"Validation samples: {len(self.val_dataloader.dataset) if hasattr(self.val_dataloader, 'dataset') else 'Unknown'}")
        logging.info(f"Training batches: {len(self.train_dataloader)}")
        logging.info(f"Validation batches: {len(self.val_dataloader)}")
        logging.info("-" * 50)
        
        # Log model architecture
        if hasattr(self.model, "_log_model_architecture"):
            self.model._log_model_architecture()
        logging.info("=" * 50) 