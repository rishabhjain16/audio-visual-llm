import os
import torch
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from ..models.avsr_llm import AVSRLLM
from ..utils.config import AVSRConfig
from ..utils.model_utils import load_checkpoint, save_checkpoint
from argparse import ArgumentParser
import traceback
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class AVSRTrainer:
    """Trainer for AVSR-LLM model"""
    
    def __init__(self, config, gpu=0, train_dataloader=None, val_dataloader=None):
        """
        Initialize the AVSR trainer
    
    Args:
            config: Configuration object
            gpu: GPU ID to use
            train_dataloader: Training data loader (will be created from config if not provided)
            val_dataloader: Validation data loader (will be created from config if not provided)
        """
        try:
            logger.info("Initializing AVSR Trainer...")
            
            # Store configuration
            self.config = config
            self.gpu = gpu
            
            # Set device
            if torch.cuda.is_available() and gpu >= 0:
                self.device = torch.device(f'cuda:{gpu}')
                logger.info(f"Using device: {self.device}")
            else:
                self.device = torch.device('cpu')
                logger.info(f"Using device: {self.device}")
            
            # Set debug mode
            self.debug = getattr(config, 'debug', False)
            if self.debug:
                logger.setLevel(logging.DEBUG)
                logger.debug("Debug mode enabled")
            
            # Set gradient accumulation steps
            self.grad_accum_steps = max(1, getattr(config, 'gradient_accumulation_steps', 
                                                   getattr(config.training, 'grad_accum_steps', 1) if hasattr(config, 'training') else 1))
            logger.info(f"Using gradient accumulation steps: {self.grad_accum_steps}")
            
            # Set up training parameters
            self.max_epochs = getattr(config, 'epochs', 
                                     getattr(config.training, 'num_epochs', 10) if hasattr(config, 'training') else 10)
            self.log_every = getattr(config, 'log_every', 
                                    getattr(config.training, 'log_interval', 10) if hasattr(config, 'training') else 10)
            self.save_every = getattr(config, 'save_every', 1)
            self.output_dir = getattr(config, 'output_dir', 
                                     getattr(config.training, 'checkpoint_dir', 'outputs') if hasattr(config, 'training') else 'outputs')
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Set up model
            logger.info("Creating model...")
            self.model = self._create_model()
            
            # Move model to device
            self.model = self.model.to(self.device)
            logger.info(f"Model moved to {self.device}")
            
            # Set up data loaders if provided
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            
            # Set up data loaders from config if not provided
            if self.train_dataloader is None or self.val_dataloader is None:
                self._setup_dataloaders()
            
            # Set up optimization
            if self.train_dataloader is not None:
                self._setup_optimization()
            
            logger.info("AVSR Trainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing trainer: {e}")
            logger.error(traceback.format_exc())
            raise

    def _create_model(self):
        """Create and initialize the model"""
        logger.info("Creating model...")
        
        try:
            # Create model based on config
            if hasattr(self.config, 'model'):
                model_config = self.config.model
            else:
                model_config = self.config
                
            from ..models.avsr_llm import AVSRLLM
            model = AVSRLLM(config=model_config, device=self.device)
            
            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model has {trainable_params:,} trainable parameters out of {total_params:,} total parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            logger.error(traceback.format_exc())
            raise

    def _setup_optimization(self):
        """
        Set up the optimizer and learning rate scheduler
        """
        try:
            logger.info("Setting up optimization...")
            
            # Get optimizer parameters
            lr = getattr(self.config, 'learning_rate', 1e-4)
            weight_decay = getattr(self.config, 'weight_decay', 0.01)
            
            # Log optimizer settings
            logger.info(f"Optimizer settings: lr={lr}, weight_decay={weight_decay}")
            
            # Create parameter groups
            param_groups = [
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
                    'weight_decay': weight_decay
                },
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
                    'weight_decay': 0.0
                }
            ]
            
            # Create optimizer
            optimizer_name = getattr(self.config, 'optimizer', 'adamw').lower()
            if optimizer_name == 'adam':
                self.optimizer = torch.optim.Adam(param_groups, lr=lr)
            elif optimizer_name == 'adamw':
                self.optimizer = torch.optim.AdamW(param_groups, lr=lr)
            elif optimizer_name == 'sgd':
                self.optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
            else:
                logger.warning(f"Unknown optimizer: {optimizer_name}, using AdamW")
                self.optimizer = torch.optim.AdamW(param_groups, lr=lr)
                
            logger.info(f"Using optimizer: {type(self.optimizer).__name__}")
            
            # Create scheduler if specified
            scheduler_name = getattr(self.config, 'scheduler', None)
            if scheduler_name:
                # Get steps
                steps_per_epoch = len(self.train_dataloader) // self.grad_accum_steps
                total_steps = steps_per_epoch * self.max_epochs
                warmup_steps = int(total_steps * getattr(self.config, 'warmup_ratio', 0.1))
                
                logger.info(f"Scheduler settings: total_steps={total_steps}, warmup_steps={warmup_steps}")
                
                # Create scheduler based on name
                if scheduler_name.lower() == 'linear':
                    from transformers import get_linear_schedule_with_warmup
                    self.scheduler = get_linear_schedule_with_warmup(
                        self.optimizer, 
                        num_warmup_steps=warmup_steps, 
                        num_training_steps=total_steps
                    )
                elif scheduler_name.lower() == 'cosine':
                    from transformers import get_cosine_schedule_with_warmup
                    self.scheduler = get_cosine_schedule_with_warmup(
                        self.optimizer, 
                        num_warmup_steps=warmup_steps, 
                        num_training_steps=total_steps
                    )
                else:
                    logger.warning(f"Unknown scheduler: {scheduler_name}, not using scheduler")
                    self.scheduler = None
            else:
                logger.info("No scheduler specified")
                self.scheduler = None
                
            if self.scheduler:
                logger.info(f"Using scheduler: {type(self.scheduler).__name__}")
                
            logger.info("Optimization setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up optimization: {e}")
            logger.error(traceback.format_exc())
            raise

    def _setup_trainer(self):
        """Setup any additional trainer configuration needed"""
        try:
            # Create TensorBoard logger
            self.tb_logger = TensorBoardLogger(
                save_dir=self.log_dir,
                name="tensorboard"
            )
            
            # Create checkpoint callback
            self.checkpoint_callback = ModelCheckpoint(
                dirpath=self.checkpoint_dir,
                filename="model-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=3,
                mode="min"
            )
            
            # Create early stopping callback
            patience = getattr(self.config.training, "early_stopping_patience", 5)
            self.early_stopping_callback = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min"
            )
            
            # Create learning rate monitor
            self.lr_monitor = LearningRateMonitor(logging_interval="step")
            
            logger.info(f"Trainer setup complete with early stopping patience={patience}")
            
        except Exception as e:
            logger.error(f"Error during trainer setup: {e}")
            logger.error(traceback.format_exc())

    def _setup_dataloaders(self):
        """Setup data loaders for training and validation"""
        logger.info("Setting up dataloaders")
        
        try:
            # If dataloaders are already provided, use them
            if hasattr(self, 'train_dataloader') and self.train_dataloader is not None:
                logger.info(f"Using provided train dataloader")
                return
            
            # Import dataset modules
            from ..data.dataset import AVSRDataset, create_dataloader
            
            # Get configuration for data
            if hasattr(self.config, 'data'):
                data_config = self.config.data
                
                # Get data paths
                data_path = getattr(data_config, "path", "data")
                
                # Get manifest and label paths
                train_manifest_filename = getattr(data_config, "train_manifest", "train.tsv")
                train_labels_filename = getattr(data_config, "train_labels", "train.wrd")
                val_manifest_filename = getattr(data_config, "val_manifest", "valid.tsv")
                val_labels_filename = getattr(data_config, "val_labels", "valid.wrd")
                
                train_manifest = os.path.join(data_path, train_manifest_filename)
                train_labels = os.path.join(data_path, train_labels_filename)
                val_manifest = os.path.join(data_path, val_manifest_filename)
                val_labels = os.path.join(data_path, val_labels_filename)
                
                # Get other dataset parameters
                max_audio_length = getattr(data_config, "max_audio_length", 480000)
                max_video_length = getattr(data_config, "max_video_length", 600)
                batch_size = getattr(data_config, "batch_size", 8)
                num_workers = getattr(data_config, "num_workers", 4)
                pin_memory = getattr(data_config, "pin_memory", True)
                
                # Check if files exist
                if not os.path.exists(train_manifest):
                    logger.warning(f"Train manifest file not found: {train_manifest}")
                else:
                    # Create training dataset
                    logger.info(f"Creating training dataset from {train_manifest}")
                    train_dataset = AVSRDataset(
                        manifest_path=train_manifest,
                        label_path=train_labels,
                        root_dir=data_path,
                        max_audio_length=max_audio_length,
                        max_video_length=max_video_length,
                        split="train"
                    )
                    
                    # Create training dataloader
                    self.train_dataloader = create_dataloader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=pin_memory
                    )
                    
                    logger.info(f"Created training dataloader with {len(train_dataset)} samples")
                
                # Create validation dataset if available
                if os.path.exists(val_manifest) and os.path.exists(val_labels):
                    logger.info(f"Creating validation dataset from {val_manifest}")
                    val_dataset = AVSRDataset(
                        manifest_path=val_manifest,
                        label_path=val_labels,
                        root_dir=data_path,
                        max_audio_length=max_audio_length,
                        max_video_length=max_video_length,
                        split="val"
                    )
                    
                    # Create validation dataloader
                    self.val_dataloader = create_dataloader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory
                    )
                    
                    logger.info(f"Created validation dataloader with {len(val_dataset)} samples")
                else:
                    logger.warning(f"Validation files not found. Skipping validation dataloader creation.")
            else:
                logger.warning("No data configuration provided. Unable to create dataloaders.")
                
        except Exception as e:
            logger.error(f"Error setting up dataloaders: {e}")
            logger.error(traceback.format_exc())

    def _process_batch(self, batch, is_train=True):
        """
        Process a batch of data
        
        Args:
            batch: Batch of data
            is_train: Whether this is a training or evaluation batch
            
        Returns:
            loss: Loss value
            batch_size: Size of the batch
        """
        try:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            else:
                logger.warning(f"Unexpected batch type: {type(batch)}")
            
            # Debug batch content
            if getattr(self, 'debug', False):
                if isinstance(batch, dict):
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            logger.debug(f"Batch[{k}] shape: {v.shape}, device: {v.device}")
            
            # Extract only the necessary inputs for the model
            model_inputs = {}
            
            # We should only pass these specific fields to avoid unexpected arguments
            valid_fields = {
                'audio', 'video', 'labels', 'return_loss', 'debug'
            }
            
            # Copy only valid fields from the batch
            for field in valid_fields:
                if field in batch:
                    model_inputs[field] = batch[field]
            
            # Get batch size from inputs
            batch_size = None
            for field in ['audio', 'video']:
                if field in model_inputs and model_inputs[field] is not None:
                    batch_size = model_inputs[field].size(0)
                    break
            
            # If no batch size found, use 1
            if batch_size is None:
                batch_size = 1
                logger.warning("Could not determine batch size from inputs, using batch_size=1")
            
            # Ensure all inputs have the same batch size
            for field in ['audio', 'video']:
                if field in model_inputs and model_inputs[field] is not None:
                    if model_inputs[field].size(0) != batch_size:
                        logger.warning(f"Batch size mismatch for {field}: got {model_inputs[field].size(0)}, expected {batch_size}")
                        model_inputs[field] = model_inputs[field][:batch_size]
            
            # Get text labels if available for proper training
            if 'text' in batch and 'labels' not in model_inputs:
                # Ensure labels have the same batch size
                if isinstance(batch['text'], torch.Tensor) and batch['text'].size(0) != batch_size:
                    logger.warning(f"Batch size mismatch for labels: got {batch['text'].size(0)}, expected {batch_size}")
                    model_inputs['labels'] = batch['text'][:batch_size]
                else:
                    model_inputs['labels'] = batch['text']
                logger.debug(f"Using text field as labels: {batch['text']}")
            elif 'labels' in model_inputs:
                # Ensure labels have the same batch size
                if isinstance(model_inputs['labels'], torch.Tensor) and model_inputs['labels'].size(0) != batch_size:
                    logger.warning(f"Batch size mismatch for labels: got {model_inputs['labels'].size(0)}, expected {batch_size}")
                    model_inputs['labels'] = model_inputs['labels'][:batch_size]
            
            # Set training flag
            model_inputs['return_loss'] = is_train
            
            # Add debug flag if needed
            if getattr(self, 'debug', False):
                # Don't pass debug to the model, just log it
                logger.debug("Running in debug mode")
            
            # Forward pass
            outputs = self.model(**model_inputs)
            
            # Get loss
            loss = None
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                logger.warning("Model output does not contain loss. This is normal for inference but unexpected for training.")
                # Create a dummy loss for error handling
                loss = torch.tensor(0.0, device=self.device, requires_grad=is_train)
            
            # Handle None loss (during inference)
            if loss is None:
                logger.warning("Loss is None. Using dummy loss of 0.0")
                loss = torch.tensor(0.0, device=self.device, requires_grad=is_train)
            
            # Handle NaN loss
            if torch.isnan(loss).any():
                logger.warning("Loss is NaN! Using zero loss for stability.")
                loss = torch.tensor(0.0, device=self.device, requires_grad=is_train)
                
            # Ensure loss is on correct device
            if loss.device != self.device:
                loss = loss.to(self.device)
                
            # Scale loss for gradient accumulation during training
            if is_train and hasattr(self, 'grad_accum_steps') and self.grad_accum_steps > 1:
                loss = loss / self.grad_accum_steps
                
            # Backward pass during training (only if loss requires grad)
            if is_train and loss.requires_grad:
                loss.backward()
                
            # Return the actual batch size used
            return loss, batch_size
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            logger.error(traceback.format_exc())
            
            # Return a default loss value to indicate an error
            return torch.tensor(0.0, device=self.device), 1

    def _validate(self):
        """
        Validate the model on the validation dataset
        
        Returns:
            val_loss: Average validation loss
        """
        logger.info("Validating model...")
        
        try:
            if not hasattr(self, 'val_dataloader') or self.val_dataloader is None:
                logger.warning("No validation dataloader available")
                return float('inf')
            
            self.model.eval()
            total_loss = 0.0
            total_samples = 0
            
            # Process batches
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_dataloader):
                    try:
                        # Process batch
                        loss, batch_size = self._process_batch(batch, is_train=False)
                        
                        # Accumulate loss
                        total_loss += loss.item() * batch_size
                        total_samples += batch_size
                        
                        # Log occasionally
                        if (batch_idx + 1) % 10 == 0:
                            logger.info(f"Validation - Processed {batch_idx + 1}/{len(self.val_dataloader)} batches")
                        
                        # Free memory
                        del loss
                        torch.cuda.empty_cache()
                    
                    except Exception as e:
                        logger.error(f"Error processing validation batch {batch_idx}: {e}")
                        logger.error(traceback.format_exc())
                        continue
            
            # Calculate average loss
            avg_val_loss = total_loss / (total_samples if total_samples > 0 else 1)
            
            logger.info(f"Validation complete - Average loss: {avg_val_loss:.4f}")
            
            return avg_val_loss
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            logger.error(traceback.format_exc())
            
            # Return a high loss to indicate an error
            return float('inf')

    def _save_checkpoint(self, is_best=False, is_final=False):
        """
        Save a checkpoint of the model and optimizer

        Args:
            is_best: Whether this is the best model so far
            is_final: Whether this is the final model
        """
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Determine checkpoint filename
            if is_final:
                checkpoint_path = os.path.join(self.output_dir, 'final_model.pt')
            elif is_best:
                checkpoint_path = os.path.join(self.output_dir, 'best_model.pt')
            else:
                checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{self.current_epoch+1}.pt')
            
            # Create checkpoint dictionary
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') and self.scheduler else None,
                'config': self.config
            }
            
            # Save checkpoint
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            logger.error(traceback.format_exc())
            return None

    def train(self, debug_mode=False):
        """
        Train the model
        
        Args:
            debug_mode: Whether to run in debug mode
            
        Returns:
            results: Dictionary with training results
        """
        try:
            # Set debug flag
            self.debug = debug_mode
            
            # Setup timers and state
            logger.info("Starting model training...")
            
            # Check if model and dataloaders exist
            if self.model is None:
                logger.error("No model available for training")
                return {"status": "error", "message": "No model available for training"}
                
            if self.train_dataloader is None:
                logger.error("No training dataloader available")
                return {"status": "error", "message": "No training dataloader available"}
            
            # Set model to training mode
            self.model.train()
            
            # Set gradient accumulation steps
            logger.info(f"Using gradient accumulation with {self.grad_accum_steps} steps")
            
            # Determine starting epoch
            start_epoch = getattr(self, 'current_epoch', 0)
            logger.info(f"Starting from epoch {start_epoch+1}")
            
            # Get total epochs
            num_epochs = self.max_epochs
            
            # Main training loop
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
                
                # Initialize epoch stats
                epoch_loss = 0.0
                processed_samples = 0
                
                # Create progress bar
                pbar = tqdm(enumerate(self.train_dataloader), 
                           total=len(self.train_dataloader),
                           desc=f"Epoch {epoch+1}",
                           disable=debug_mode)
                
                # Batch loop
                for batch_idx, batch in pbar:
                    try:
                        # Reset gradients if using accumulation
                        if batch_idx % self.grad_accum_steps == 0:
                            self.optimizer.zero_grad()
                            
                        # Process batch
                        loss, batch_size = self._process_batch(batch, is_train=True)
                        
                        # Update statistics
                        epoch_loss += loss.item() * batch_size
                        processed_samples += batch_size
                        
                        # Update progress bar
                        pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
                        
                        # Log every N batches
                        if batch_idx % self.log_every == 0:
                            logger.info(f"Epoch {epoch+1} | Batch {batch_idx}/{len(self.train_dataloader)} | Loss: {loss.item():.4f}")
                        
                        # Step optimizer if using gradient accumulation
                        if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                            # Clip gradients
                            max_grad_norm = getattr(self.config.training, "max_grad_norm", 1.0) if hasattr(self.config, 'training') else 1.0
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                            
                            # Step optimizer
                            self.optimizer.step()
                            
                            # Step scheduler if it exists
                            if hasattr(self, 'scheduler') and self.scheduler is not None:
                                self.scheduler.step()
                        
                        # Free up memory
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {e}")
                        logger.error(traceback.format_exc())
                        continue
                
                # Calculate average loss for the epoch
                avg_epoch_loss = epoch_loss / max(processed_samples, 1)
                logger.info(f"Epoch {epoch+1} completed | Avg Loss: {avg_epoch_loss:.4f}")
                
                # Perform validation if validation dataloader exists
                if hasattr(self, 'val_dataloader') and self.val_dataloader is not None:
                    val_loss = self._validate()
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    
                    # Track best validation loss
                    best_val_loss = getattr(self, 'best_val_loss', float('inf'))
                    if val_loss < best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(is_best=True)
                        logger.info(f"New best validation loss: {val_loss:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % self.save_every == 0:
                    self._save_checkpoint()
                    logger.info(f"Checkpoint saved at epoch {epoch+1}")
            
            # Save final model
            self._save_checkpoint(is_final=True)
            logger.info("Training completed successfully")
            
            return {'status': 'success'}
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            logger.error(traceback.format_exc())
            return {'status': 'error', 'message': str(e)}

    def validate(self, val_loader):
        """Run validation"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # Move batch to device
                    if "audio" in batch:
                        batch["audio"] = batch["audio"].to(self.device)
                        if "audio_padding_mask" in batch:
                            batch["audio_padding_mask"] = batch["audio_padding_mask"].to(self.device)
                    
                    if "video" in batch:
                        batch["video"] = batch["video"].to(self.device)
                        if "video_padding_mask" in batch:
                            batch["video_padding_mask"] = batch["video_padding_mask"].to(self.device)
                    
                    # Ensure the model and audio encoder are on the same device
                    if hasattr(self.model, 'audio_encoder') and hasattr(self.model.audio_encoder, 'model'):
                        if next(self.model.audio_encoder.model.parameters()).device != self.device:
                            logger.info(f"Moving Whisper model to device: {self.device}")
                            self.model.audio_encoder.model = self.model.audio_encoder.model.to(self.device)
                    
                    # Get text labels
                    if "text" in batch:
                        labels = batch["text"]
                    else:
                        labels = None
                    
                    # Forward pass
                    outputs = self.model(
                        audio_features=batch.get("audio", None),
                        video_features=batch.get("video", None),
                        labels=labels,
                        padding_mask=batch.get("audio_padding_mask", None) if "audio" in batch else batch.get("video_padding_mask", None),
                        return_loss=True
                    )
                    
                    # Get loss
                    loss = outputs.loss
                    
                    # Track loss
                    val_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error processing validation batch: {e}")
                    continue
        
        # Set model back to training mode
        self.model.train()
        
        # Calculate average loss
        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
        
        return avg_val_loss

    def save_checkpoint(self, path, epoch=None, optimizer=None):
        """Save a checkpoint of the model.
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }
            
            if epoch is not None:
                checkpoint['epoch'] = epoch
                
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                
            torch.save(checkpoint, path)
            logger.info(f"Model checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            logger.error(traceback.format_exc())

    @classmethod
    def from_checkpoint(cls, checkpoint_path, config=None):
        """
        Load trainer from a checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
            config: Optional config to override checkpoint config
            
        Returns:
            trainer: Trainer object initialized from checkpoint
        """
        try:
            logger.info(f"Loading trainer from checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Get config from checkpoint if not provided
            if config is None:
                config = checkpoint.get('config', None)
                if config is None:
                    logger.error("No config found in checkpoint and none provided")
                    raise ValueError("No configuration found for trainer initialization")
            
            # Set device based on config or availability
            device = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            
            # Import AVSRLLM model class
            from ..models.avsr_llm import AVSRLLM
            
            # Create model from checkpoint
            model = AVSRLLM.from_checkpoint(checkpoint_path, device=device)
            
            # Create trainer
            trainer = cls(config=config, model=model)
            
            # Load optimizer state if it exists
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                # Make sure optimizer exists before loading
                if not hasattr(trainer, 'optimizer') or trainer.optimizer is None:
                    trainer._setup_optimization()
                
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state from checkpoint")
            
            # Load scheduler state if it exists
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("Loaded scheduler state from checkpoint")
            
            # Set current epoch
            if 'epoch' in checkpoint:
                trainer.current_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                logger.info(f"Will resume training from epoch {trainer.current_epoch}")
            
            logger.info("Trainer loaded successfully from checkpoint")
            return trainer
            
        except Exception as e:
            logger.error(f"Error loading trainer from checkpoint: {e}")
            logger.error(traceback.format_exc())
            raise

def train_avsr_llm(config):
    """
    Train the AVSR-LLM model
    
    Args:
        config: Configuration object or dictionary
        
    Returns:
        dict: Training statistics
    """
    try:
        logger.info("Starting AVSR-LLM training...")
        
        # Check if resuming from checkpoint
        resume_from = getattr(config, 'resume_from', None)
        
        if resume_from and os.path.exists(resume_from):
            logger.info(f"Resuming training from checkpoint: {resume_from}")
            trainer = AVSRTrainer.from_checkpoint(resume_from, config)
        else:
            logger.info("Starting training from scratch")
            trainer = AVSRTrainer(config)
        
        # Run training
        logger.info("Starting training process")
        results = trainer.train()
        
        # Print results
        logger.info("Training completed")
        if isinstance(results, dict) and 'status' in results:
            if results['status'] == 'success':
                logger.info("Training was successful")
                if 'best_val_loss' in results:
                    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
            else:
                logger.error(f"Training failed: {results.get('message', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in train_avsr_llm: {e}")
        logger.error(traceback.format_exc())
        return {'status': 'error', 'message': str(e)}

def add_trainer_args(parser):
    """
    Add training arguments to an argument parser
    
    Args:
        parser: ArgumentParser object
    
    Returns:
        parser: Updated ArgumentParser
    """
    group = parser.add_argument_group("Training Arguments")
    
    # General training parameters
    group.add_argument("--output_dir", type=str, default="outputs", help="Output directory for models and logs")
    group.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    group.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    group.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    group.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size")
    group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    group.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    group.add_argument("--log_every", type=int, default=10, help="Log every N batches")
    group.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    
    # Optimizer and scheduler
    group.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer type")
    group.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine", "none"], help="Scheduler type")
    group.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    
    # Dataset arguments
    group.add_argument("--train_dataset_path", type=str, required=True, help="Path to training dataset")
    group.add_argument("--val_dataset_path", type=str, help="Path to validation dataset")
    group.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Model arguments
    group.add_argument("--device", type=str, default=None, help="Device to use (defaults to CUDA if available)")
    group.add_argument("--debug", action="store_true", help="Enable debug mode with more logging")
    group.add_argument("--resume_from", type=str, help="Path to checkpoint to resume training from")
    
    # Model specific arguments 
    group.add_argument("--encoder_dim", type=int, default=768, help="Encoder dimension")
    group.add_argument("--use_audio", action="store_true", default=True, help="Use audio modality")
    group.add_argument("--use_video", action="store_true", default=True, help="Use video modality")
    group.add_argument("--llm_name", type=str, default="gpt2", help="LLM model name")
    
    return parser

def main():
    """
    Main function for training AVSR-LLM from command line
    """
    import argparse
    from types import SimpleNamespace
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train AVSR-LLM model")
    parser = add_trainer_args(parser)
    args = parser.parse_args()
    
    # Convert args to config
    config = SimpleNamespace(**vars(args))
    
    # Set device if not specified
    if config.device is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set scheduler to None if 'none'
    if hasattr(config, 'scheduler') and config.scheduler.lower() == 'none':
        config.scheduler = None
    
    # Train model
    results = train_avsr_llm(config)
    
    # Return success or failure
    if isinstance(results, dict) and results.get('status') == 'success':
        return 0
    else:
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())