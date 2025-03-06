import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR
import numpy as np
import argparse
from pathlib import Path
import json
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter

from ..models.avsr_llm import AVSRLLM
from ..data.dataset import AVSRDataset, create_dataloader

class Trainer:
    """Trainer for AVSR-LLM models"""
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        optimizer=None,
        scheduler=None,
        device="cuda",
        output_dir="./output",
        max_epochs=10,
        log_interval=10,
        save_interval=1,
        grad_accum_steps=1,
        clip_grad_norm=1.0,
        eval_steps=None,
        best_metric="wer",
        lower_is_better=True,
    ):
        """
        Args:
            model: AVSRLLM model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            output_dir: Directory to save checkpoints and logs
            max_epochs: Maximum number of epochs
            log_interval: Interval to log training progress
            save_interval: Interval to save checkpoints
            grad_accum_steps: Gradient accumulation steps
            clip_grad_norm: Gradient clipping norm
            eval_steps: Evaluate every N steps (None for once per epoch)
            best_metric: Metric to use for best model selection
            lower_is_better: Whether lower metric is better
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm = clip_grad_norm
        self.eval_steps = eval_steps
        self.best_metric = best_metric
        self.lower_is_better = lower_is_better
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(output_dir, "logs"))
        
        # Initialize tracking variables
        self.global_step = 0
        self.epoch = 0
        self.best_metric_value = float("inf") if lower_is_better else float("-inf")
    
    def train(self):
        """Train the model"""
        print(f"Starting training for {self.max_epochs} epochs")
        
        # Move model to device
        self.model.to(self.device)
        
        # Training loop
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            print(f"
Epoch {epoch+1}/{self.max_epochs}")
            
            # Train for one epoch
            self.train_epoch()
            
            # Evaluate if validation set is available
            if self.val_dataloader is not None:
                print("
Running validation...")
                metrics = self.evaluate()
                
                # Log metrics
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"val/{name}", value, self.global_step)
                
                # Save best model
                if self.best_metric in metrics:
                    metric_value = metrics[self.best_metric]
                    is_best = (self.lower_is_better and metric_value < self.best_metric_value) or \n                              (not self.lower_is_better and metric_value > self.best_metric_value)
                    
                    if is_best:
                        self.best_metric_value = metric_value
                        self.save_checkpoint("best")
            
            # Save checkpoint every save_interval epochs
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")
        
        # Save final model
        self.save_checkpoint("final")
        print("
Training complete")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Progress bar
        progress = tqdm(total=len(self.train_dataloader), desc=f"Epoch {self.epoch+1}")
        
        # Tracking variables
        epoch_loss = 0.0
        steps = 0
        
        # Iterate over batches
        for i, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            loss = outputs.loss / self.grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if gradient accumulation is complete
            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.train_dataloader):
                # Clip gradients
                if self.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # Update tracking variables
            epoch_loss += loss.item() * self.grad_accum_steps
            steps += 1
            self.global_step += 1
            
            # Update progress bar
            progress.update(1)
            progress.set_postfix({"loss": loss.item() * self.grad_accum_steps})
            
            # Log every log_interval steps
            if steps % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/loss", loss.item() * self.grad_accum_steps, self.global_step)
                self.writer.add_scalar("train/lr", lr, self.global_step)
            
            # Evaluate if eval_steps is set
            if self.eval_steps is not None and self.global_step % self.eval_steps == 0:
                if self.val_dataloader is not None:
                    metrics = self.evaluate()
                    
                    # Log metrics
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f"val/{name}", value, self.global_step)
                    
                    # Return to training mode
                    self.model.train()
        
        # Close progress bar
        progress.close()
        
        # Calculate epoch loss
        epoch_loss /= steps
        print(f"Epoch {self.epoch+1} loss: {epoch_loss:.4f}")
    
    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        eval_loss = 0.0
        steps = 0
        
        # Iterate over validation batches
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Labels for computing loss
                labels = batch.get("text", None)
                
                if labels is not None:
                    # Forward pass with loss computation
                    outputs = self.model(batch)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    steps += 1
                
                # Generate predictions
                preds = self.model.generate(batch)
                
                # Store predictions and labels
                all_preds.extend(preds)
                if labels is not None:
                    all_labels.extend(labels)
        
        # Calculate metrics
        metrics = {}
        
        if steps > 0:
            metrics["loss"] = eval_loss / steps
        
        if all_labels:
            # Calculate model-specific metrics
            model_metrics = self.model.calculate_metrics(all_preds, all_labels)
            metrics.update(model_metrics)
        
        # Print metrics
        print("Validation metrics:")
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
        
        # Save some example predictions
        if all_preds and all_labels:
            examples = min(5, len(all_preds))
            print("
Example predictions:")
            for i in range(examples):
                print(f"  Reference: {all_labels[i]}")
                print(f"  Prediction: {all_preds[i]}")
                print()
        
        return metrics
    
    def save_checkpoint(self, suffix=""):
        """Save a checkpoint"""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint path
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{suffix}.pt")
        
        # Save model weights
        self.model.save_pretrained(os.path.join(checkpoint_dir, f"model_{suffix}"))
        
        # Save training state
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric_value": self.best_metric_value,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore training state
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metric_value = checkpoint["best_metric_value"]
        
        # Restore optimizer state
        if checkpoint["optimizer_state"] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Restore scheduler state
        if checkpoint["scheduler_state"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch+1}, global step {self.global_step}")
        
        return True

def create_optimizer(model, learning_rate=5e-5, weight_decay=0.01):
    """Create optimizer for model"""
    # Create parameter groups
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                      if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    return optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

def create_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    """Create learning rate scheduler"""
    # Calculate warmup steps
    warmup_steps = int(num_training_steps * warmup_ratio)
    
    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Create main scheduler
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - warmup_steps,
        eta_min=1e-6
    )
    
    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train AVSR-LLM model")
    
    # Model arguments
    parser.add_argument("--avhubert_path", type=str, required=True,
                        help="Path to AV-HuBERT checkpoint")
    parser.add_argument("--llm_path", type=str, required=True,
                        help="Path to LLM model")
    parser.add_argument("--use_audio", action="store_true",
                        help="Use audio modality")
    parser.add_argument("--use_video", action="store_true",
                        help="Use video modality")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder parameters")
    parser.add_argument("--encoder_finetune_layers", type=int, nargs="*",
                        help="List of encoder layers to finetune")
    parser.add_argument("--encoder_layer", type=int, default=-1,
                        help="Encoder layer to extract features from")
    
    # LLM arguments
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for fine-tuning")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--prompt_template", type=str, default="Transcribe the speech: ",
                        help="Prompt template for LLM")
    
    # Data arguments
    parser.add_argument("--train_manifest", type=str, required=True,
                        help="Path to training manifest file")
    parser.add_argument("--train_labels", type=str, required=True,
                        help="Path to training labels file")
    parser.add_argument("--val_manifest", type=str,
                        help="Path to validation manifest file")
    parser.add_argument("--val_labels", type=str,
                        help="Path to validation labels file")
    parser.add_argument("--data_root", type=str,
                        help="Root directory for data paths")
    parser.add_argument("--max_audio_length", type=int, default=480000,
                        help="Maximum audio length in samples")
    parser.add_argument("--max_video_length", type=int, default=600,
                        help="Maximum video length in frames")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log interval")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Save interval (epochs)")
    parser.add_argument("--eval_steps", type=int,
                        help="Evaluate every N steps")
    parser.add_argument("--resume_from", type=str,
                        help="Resume from checkpoint")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Create datasets
    print(f"Loading training data from {args.train_manifest}")
    train_dataset = AVSRDataset(
        manifest_path=args.train_manifest,
        label_path=args.train_labels,
        root_dir=args.data_root,
        modalities=[m for m in ["audio", "video"] if getattr(args, f"use_{m}")],
        max_audio_length=args.max_audio_length,
        max_video_length=args.max_video_length,
        split="train"
    )
    
    val_dataset = None
    if args.val_manifest and args.val_labels:
        print(f"Loading validation data from {args.val_manifest}")
        val_dataset = AVSRDataset(
            manifest_path=args.val_manifest,
            label_path=args.val_labels,
            root_dir=args.data_root,
            modalities=[m for m in ["audio", "video"] if getattr(args, f"use_{m}")],
            max_audio_length=args.max_audio_length,
            max_video_length=args.max_video_length,
            split="val"
        )
    
    # Create data loaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    # Create model
    print("Initializing model")
    model = AVSRLLM(
        avhubert_path=args.avhubert_path,
        llm_path=args.llm_path,
        use_audio=args.use_audio,
        use_video=args.use_video,
        freeze_encoder=args.freeze_encoder,
        encoder_finetune_layers=args.encoder_finetune_layers,
        encoder_layer=args.encoder_layer,
        use_lora=args.use_lora,
        use_8bit=args.use_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        prompt_template=args.prompt_template,
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    total_steps = len(train_dataloader) // args.grad_accum_steps * args.max_epochs
    scheduler = create_scheduler(
        optimizer,
        num_training_steps=total_steps,
        warmup_ratio=args.warmup_ratio
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        grad_accum_steps=args.grad_accum_steps,
        clip_grad_norm=args.clip_grad_norm,
        eval_steps=args.eval_steps,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main()