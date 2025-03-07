import os
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

class SimpleTrainer:
    """
    A simplified trainer for the AVSR model
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_epochs=10,
        output_dir="outputs/simple_avsr",
        device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=False,
        grad_accum_steps=4,
        log_interval=10,
        save_interval=1,
        max_grad_norm=0.5,
        warmup_ratio=0.1,
        logging_steps=10
    ):
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
        self.save_interval = save_interval
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up optimizer with more stable settings
        self._setup_optimizer()
        
        # Log initialization
        logging.info(f"Initialized SimpleTrainer with:"
                   f"\n  Learning rate: {learning_rate}"
                   f"\n  Weight decay: {weight_decay}"
                   f"\n  Max epochs: {max_epochs}"
                   f"\n  Output dir: {output_dir}"
                   f"\n  Device: {device}"
                   f"\n  FP16: {fp16}"
                   f"\n  Gradient accumulation steps: {grad_accum_steps}"
                   f"\n  Max grad norm: {max_grad_norm}"
                   f"\n  Warmup ratio: {warmup_ratio}")
        
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
        """Setup optimizer and learning rate scheduler"""
        # Get trainable parameters
        trainable_params = []
        no_decay_params = []
        
        # Log which parts of the model have trainable parameters
        no_grad_count = 0
        trainable_count = 0
        
        # First collect all trainable parameters
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                no_grad_count += param.numel()
                continue
            
            # Track parameters without weight decay (bias, LayerNorm)
            if 'bias' in name or 'norm' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                trainable_params.append(param)
            
            trainable_count += param.numel()
        
        logging.info(f"Total parameters: {no_grad_count + trainable_count:,}")
        logging.info(f"Trainable parameters: {trainable_count:,} ({100 * trainable_count / (no_grad_count + trainable_count):.2f}%)")
        
        # Create parameter groups
        optim_groups = [
            {"params": trainable_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        # Use a smaller initial learning rate
        initial_lr = self.learning_rate * 0.1
        
        # Create optimizer
        self.optimizer = AdamW(
            optim_groups,
            lr=initial_lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Calculate total steps
        num_training_steps = len(self.train_dataloader) * self.max_epochs // self.grad_accum_steps
        
        # Create scheduler with warmup
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logging.info(f"Optimizer: AdamW with LR: {initial_lr} -> {self.learning_rate}")
        logging.info(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
        logging.info(f"Scheduler: Linear warmup with decay")
    
    def train(self):
        """Train the model"""
        # Track loss
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(self.max_epochs):
            logging.info(f"Starting epoch {epoch+1}/{self.max_epochs}")
            
            # Train
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            if self.val_dataloader is not None:
                val_loss = self._validate()
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(is_best=True)
                    logging.info(f"New best validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint()
            
            # Log progress
            if self.val_dataloader is not None:
                logging.info(f"Epoch {epoch+1}/{self.max_epochs} - "
                           f"Train loss: {train_loss:.4f}, "
                           f"Val loss: {val_loss:.4f}")
            else:
                logging.info(f"Epoch {epoch+1}/{self.max_epochs} - "
                           f"Train loss: {train_loss:.4f}")
        
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
                desc = f"Epoch {epoch+1} | Loss: {loss:.4f}"
                if llm_frozen:
                    desc += " (LLM frozen)"
                pbar.set_description(desc)
                
                # Update statistics
                batch_size = len(batch["audio"])
                total_loss += loss * batch_size
                total_samples += batch_size
                
                # Log progress
                if batch_idx % self.log_interval == 0:
                    logging.info(f"Epoch {epoch+1} | Batch {batch_idx}/{len(self.train_dataloader)} | Loss: {loss:.4f}")
                
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
                    pbar.set_description(f"Validation | Loss: {loss:.4f}")
                    
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
        
        return avg_loss
    
    def _process_batch(self, batch, batch_idx=0, is_train=True):
        """Process a batch of data"""
        try:
            # Move batch to device and check for NaN inputs
            audio = batch["audio"].to(self.device)
            video = batch["video"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Check for NaN/Inf values in input
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                logging.error(f"NaN/Inf in audio input at batch {batch_idx}")
                audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            if torch.isnan(video).any() or torch.isinf(video).any():
                logging.error(f"NaN/Inf in video input at batch {batch_idx}")
                video = torch.nan_to_num(video, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Log shapes for debugging
            logging.debug(f"Audio shape: {audio.shape}, Video shape: {video.shape}, Labels shape: {labels.shape}")
            
            # Forward pass with gradient handling
            if is_train:
                # Zero gradients for first accumulation step
                if batch_idx % self.grad_accum_steps == 0:
                    self.optimizer.zero_grad()
                
                # Forward pass with gradient computation
                outputs = self.model(
                    audio=audio,
                    video=video,
                    labels=labels,
                    return_loss=True,
                )
            else:
                # Validation pass without gradients
                with torch.no_grad():
                    outputs = self.model(
                        audio=audio,
                        video=video,
                        labels=labels,
                        return_loss=True,
                    )
            
            # Get loss with careful error handling
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
                    
                    # Clip gradients more aggressively for stability
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.max_grad_norm
                    )
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logging.error(f"NaN/Inf gradient norm at batch {batch_idx}")
                        self.optimizer.zero_grad()
                        return 1.0
                    
                    # Log gradient norm occasionally
                    if batch_idx % 100 == 0:
                        logging.info(f"Batch {batch_idx} gradient norm: {grad_norm:.4f}")
                    
                    # Step optimizer and scheduler if needed
                    if (batch_idx + 1) % self.grad_accum_steps == 0 or \
                       (batch_idx + 1) == len(self.train_dataloader):
                        self.optimizer.step()
                        self.scheduler.step()
                        
                except RuntimeError as e:
                    logging.error(f"Runtime error in backward pass: {e}")
                    self.optimizer.zero_grad()
                    return 1.0
            
            # Return loss value
            loss_value = loss.item()
            if np.isnan(loss_value) or np.isinf(loss_value):
                logging.warning(f"NaN/Inf in final loss value: {loss_value}")
                return 1.0
            
            return loss_value
            
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            logging.error(traceback.format_exc())
            return 1.0  # Return a safe value
    
    def _save_checkpoint(self, is_best=False, is_final=False):
        """Save a checkpoint"""
        # Determine path
        if is_final:
            path = os.path.join(self.output_dir, "final_model")
        elif is_best:
            path = os.path.join(self.output_dir, "best_model")
        else:
            path = os.path.join(self.output_dir, f"model_epoch_{epoch+1}")
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save optimizer and scheduler
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            },
            os.path.join(path, "optimizer.pt")
        )
        
        logging.info(f"Saved checkpoint to {path}")
    
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