#!/usr/bin/env python3
"""
Trainer for AVHuBERT-Whisper model
"""

import os
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from datetime import datetime
import json
import traceback

class AVHuBERTWhisperTrainer:
    """
    A trainer for the AVHuBERT-Whisper model that supports multiple modalities.
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_epochs=10,
        output_dir="outputs/avhubert_whisper",
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
            model: The AVHuBERTWhisperModel to train
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            learning_rate: Learning rate
            weight_decay: Weight decay
            max_epochs: Maximum number of epochs to train
            output_dir: Directory to save outputs
            device: Device to train on
            fp16: Whether to use mixed precision
            grad_accum_steps: Gradient accumulation steps
            log_interval: Log every N batches
            save_every: Save checkpoint every N epochs
            save_steps: Save checkpoint every N steps (overrides save_every)
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps
            log_param_updates: Whether to log parameter updates
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
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize best validation loss to infinity
        self.best_val_loss = float('inf')
        
        # Initialize step counter
        self.global_step = 0
        
        # Log initialization
        self._log_init()
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Number of trainable parameters: {trainable_params:,}")
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        total_steps = len(self.train_dataloader) * self.max_epochs // self.grad_accum_steps
        
        # Use CosineAnnealingLR as scheduler
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        return scheduler
    
    def _log_init(self):
        """Log initialization parameters"""
        logging.info("=" * 50)
        logging.info("Initializing AVHuBERT-Whisper Trainer")
        logging.info("=" * 50)
        
        # Log model info
        logging.info(f"Model modality: {self.model.modality}")
        logging.info(f"Model using fp16: {self.fp16}")
        
        # Log training parameters
        logging.info(f"Learning rate: {self.learning_rate}")
        logging.info(f"Weight decay: {self.weight_decay}")
        logging.info(f"Max epochs: {self.max_epochs}")
        logging.info(f"Gradient accumulation steps: {self.grad_accum_steps}")
        logging.info(f"Max gradient norm: {self.max_grad_norm}")
        
        # Log dataloaders
        if self.train_dataloader:
            logging.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        if self.val_dataloader:
            logging.info(f"Validation samples: {len(self.val_dataloader.dataset)}")
            
        logging.info("=" * 50)
    
    def train(self):
        """Train the model"""
        logging.info("Starting training...")
        
        try:
            for epoch in range(self.max_epochs):
                logging.info(f"Epoch {epoch+1}/{self.max_epochs}")
                
                # Train one epoch
                train_loss = self._train_epoch(epoch)
                
                # Validate
                val_loss = None
                if self.val_dataloader:
                    val_loss = self._validate()
                    
                # Save checkpoint
                if self.save_every and (epoch + 1) % self.save_every == 0:
                    self._save_checkpoint(epoch, train_loss, val_loss)
                    
                # Log epoch results
                logging.info(f"Epoch {epoch+1} complete: train_loss={train_loss:.4f}" + 
                            (f", val_loss={val_loss:.4f}" if val_loss is not None else ""))
                
                # Save if best model
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, train_loss, val_loss, is_best=True)
                    
            logging.info("Training completed!")
            
        except Exception as e:
            logging.error(f"Error during training: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Process batch
            loss = self._process_batch(batch, batch_idx, is_train=True)
            
            # Accumulate stats
            batch_size = batch["audio"].size(0) if "audio" in batch else batch["video"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Increment global step counter
            self.global_step += 1
            
            # Save checkpoint if needed
            if self.save_steps and self.global_step % self.save_steps == 0:
                avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
                self._save_checkpoint(epoch, avg_loss, None, step=self.global_step)
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return avg_loss
    
    def _validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Validation")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Process batch
                loss = self._process_batch(batch, batch_idx, is_train=False)
                
                # Accumulate stats
                batch_size = batch["audio"].size(0) if "audio" in batch else batch["video"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Update progress bar
                progress_bar.set_postfix({"val_loss": loss.item()})
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return avg_loss
    
    def _process_batch(self, batch, batch_idx=0, is_train=True):
        """Process a batch of data"""
        try:
            # Move batch to device and check for NaN inputs
            audio = batch["audio"].to(self.device) if "audio" in batch else None
            video = batch["video"].to(self.device) if "video" in batch else None
            labels = batch["labels"].to(self.device) if "labels" in batch else None
            prompt = batch["prompt"].to(self.device) if "prompt" in batch else None
            
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
                    prompt=prompt,
                    return_loss=True,
                )
                
                # Get loss and scale by gradient accumulation steps
                loss = outputs.loss / self.grad_accum_steps
                
                # Backward pass
                loss.backward()
                
                # Perform optimization step if accumulated enough gradients
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # Clip gradients
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.max_grad_norm
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Scheduler step
                    if self.scheduler:
                        self.scheduler.step()
                
            else:
                # Validation pass without gradients
                outputs = self.model(
                    audio=audio,
                    video=video,
                    labels=labels,
                    prompt=prompt,
                    return_loss=True,
                )
                loss = outputs.loss
            
            return loss
            
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def _save_checkpoint(self, epoch, train_loss, val_loss=None, is_best=False, step=None):
        """Save a checkpoint"""
        save_dir = self.output_dir
        
        # Create filename based on epoch or step
        if step is not None:
            filename = f"checkpoint_step_{step}"
        else:
            filename = f"checkpoint_epoch_{epoch+1}"
            
        if is_best:
            filename = "best_model"
        
        # Create save path
        save_path = os.path.join(save_dir, filename)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save optimizer and scheduler state
        torch.save({
            'epoch': epoch,
            'step': self.global_step,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }, os.path.join(save_path, "trainer_state.pt"))
        
        logging.info(f"Saved checkpoint to {save_path}")
        
        return save_path
        
    def resume_from_checkpoint(self, checkpoint_path):
        """Resume training from a checkpoint"""
        try:
            logging.info(f"Resuming from checkpoint: {checkpoint_path}")
            
            # Load trainer state
            trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")
            if os.path.exists(trainer_state_path):
                state = torch.load(trainer_state_path, map_location=self.device)
                
                # Load optimizer and scheduler state
                self.optimizer.load_state_dict(state['optimizer'])
                if self.scheduler and 'scheduler' in state:
                    self.scheduler.load_state_dict(state['scheduler'])
                
                # Load other state variables
                self.global_step = state.get('step', 0)
                self.best_val_loss = state.get('best_val_loss', float('inf'))
                
                logging.info(f"Resumed training from epoch {state.get('epoch', 0)+1}, " +
                            f"step {self.global_step}")
                
                return state.get('epoch', 0) + 1
            else:
                logging.warning(f"No trainer state found at {trainer_state_path}")
                return 0
                
        except Exception as e:
            logging.error(f"Error resuming from checkpoint: {e}")
            logging.error(traceback.format_exc())
            return 0 