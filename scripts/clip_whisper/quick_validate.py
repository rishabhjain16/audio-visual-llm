#!/usr/bin/env python3
import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import load_config
from src.clip_whisper.models.clip_whisper_model import ClipWhisperModel
from src.clip_whisper.data.simple_dataset import AVSRDataset
from src.clip_whisper.data.dataset import AVSRDataCollator
from transformers import WhisperProcessor, CLIPProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Quick validation for ClipWhisperModel")
    parser.add_argument("--config", type=str, default="configs/clip_whisper.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/quick_validate",
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to model checkpoint to load for validation")
    parser.add_argument("--subset_size", type=int, default=10,
                        help="Number of validation samples to use (smaller = faster)")
    parser.add_argument("--modality", type=str, default="both", 
                        choices=["audio", "video", "both"],
                        help="Which modalities to use")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with additional logging")
    return parser.parse_args()

def simple_collate_fn(batch):
    """A simpler collation function that works with tuple returns from dataset"""
    # Filter None entries (failed samples)
    batch = [sample for sample in batch if sample is not None]
    
    if not batch:
        return None
    
    # Unpack the tuples
    audio_features = [sample[0] for sample in batch if sample[0] is not None]
    video_features = [sample[1] for sample in batch if sample[1] is not None]
    texts = [sample[2] for sample in batch]
    
    # Handle tokens - fourth element might be different types
    tokens = []
    for sample in batch:
        if len(sample) > 3 and sample[3] is not None:
            if isinstance(sample[3], torch.Tensor):
                tokens.append(sample[3])
            elif isinstance(sample[3], list):
                tokens.append(torch.tensor(sample[3]))
            else:
                tokens.append(sample[3])
    
    # Create collated batch
    collated_batch = {}
    
    # Process audio if available
    if audio_features:
        try:
            collated_batch["audio"] = torch.stack(audio_features)
        except Exception as e:
            logging.warning(f"Could not stack audio features: {e}")
            # Try converting to tensors first
            audio_tensors = [torch.tensor(af) if not isinstance(af, torch.Tensor) else af for af in audio_features]
            collated_batch["audio"] = torch.stack(audio_tensors)
    else:
        collated_batch["audio"] = None
    
    # Process video if available
    if video_features:
        try:
            collated_batch["video"] = torch.stack(video_features)
        except Exception as e:
            logging.warning(f"Could not stack video features: {e}")
            # Try converting to tensors first
            video_tensors = [torch.tensor(vf) if not isinstance(vf, torch.Tensor) else vf for vf in video_features]
            collated_batch["video"] = torch.stack(video_tensors)
    else:
        collated_batch["video"] = None
    
    # Save text as text (not used in model forward)
    collated_batch["text"] = texts
    
    # Process tokens if available - add as labels (used in model forward)
    if tokens:
        try:
            if all(isinstance(t, torch.Tensor) for t in tokens):
                # Pad token tensors to same length
                max_len = max(t.size(0) for t in tokens)
                padded_tokens = []
                for t in tokens:
                    if t.size(0) < max_len:
                        padding = torch.ones(max_len - t.size(0), dtype=t.dtype) * -100
                        padded_tokens.append(torch.cat([t, padding]))
                    else:
                        padded_tokens.append(t)
                collated_batch["labels"] = torch.stack(padded_tokens)
            else:
                # Handle non-tensor case
                collated_batch["labels"] = tokens
        except Exception as e:
            logging.warning(f"Could not process token tensors: {e}")
            collated_batch["labels"] = tokens
    
    return collated_batch

def create_validation_loader(config, data_path, subset_size=10, batch_size=None):
    """Create a validation dataloader with a subset of data"""
    # Load processors
    logging.info("Loading Whisper and CLIP processors...")
    whisper_processor = WhisperProcessor.from_pretrained(config.model.whisper_model)
    clip_processor = CLIPProcessor.from_pretrained(config.model.clip_model)
    
    # Initialize tokenizer (use the same as in model)
    try:
        from transformers import AutoTokenizer
        logging.info(f"Loading tokenizer from {config.model.llm_path}...")
        tokenizer = AutoTokenizer.from_pretrained(config.model.llm_path)
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        logging.warning("Using a simple dummy tokenizer instead")
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
    
    # Create validation dataset
    logging.info("Creating validation dataset...")
    val_dataset = AVSRDataset(
        manifest_path=os.path.join(data_path, config.data.val_manifest),
        label_path=os.path.join(data_path, config.data.val_labels),
        root_dir=data_path,
        whisper_processor=whisper_processor,
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        max_audio_length=config.data.max_audio_length,
        max_video_length=config.data.max_video_length,
        modality=config.model.modality,
        split="val"
    )
    
    # Create a subset of the validation dataset
    if subset_size and subset_size > 0:
        total_samples = len(val_dataset)
        subset_size = min(subset_size, total_samples)
        subset_indices = torch.randperm(total_samples)[:subset_size].tolist()
        
        logging.info(f"Creating subset with {subset_size} samples out of {total_samples}")
        
        # Use overridden batch size if provided
        actual_batch_size = batch_size if batch_size is not None else config.data.batch_size
        logging.info(f"Using batch size: {actual_batch_size}")
        
        # Create dataloader with subset sampler and our simple collator
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=actual_batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(subset_indices),
            num_workers=0,  # Use 0 workers to avoid multiprocessing issues
            collate_fn=simple_collate_fn,
            pin_memory=True
        )
        
        return val_dataloader, tokenizer
    else:
        logging.warning("No subset size specified, using full validation set")
        return None, tokenizer

def simple_validate(model, val_dataloader, device="cuda"):
    """Run validation without using the trainer"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(
        enumerate(val_dataloader),
        total=len(val_dataloader),
        desc="Validation"
    )
    
    # Validation loop
    with torch.no_grad():
        for batch_idx, batch in pbar:
            try:
                # Skip None batches (from collate_fn)
                if batch is None:
                    logging.warning("Received None batch, skipping")
                    continue
                
                # Debug batch info
                logging.debug(f"Batch {batch_idx} keys: {batch.keys()}")
                
                if "audio" in batch and batch["audio"] is not None:
                    logging.debug(f"Audio shape: {batch['audio'].shape}")
                else:
                    logging.debug("No audio in batch")
                    
                if "video" in batch and batch["video"] is not None:
                    logging.debug(f"Video shape: {batch['video'].shape}")
                else:
                    logging.debug("No video in batch")
                    
                if "labels" in batch:
                    if isinstance(batch["labels"], torch.Tensor):
                        logging.debug(f"Labels shape: {batch['labels'].shape}")
                    else:
                        logging.debug(f"Labels type: {type(batch['labels'])}")
                else:
                    logging.debug("No labels in batch")
                
                # Process batch
                batch_dict = process_batch(batch, model, device)
                
                # Forward pass - with extra error handling
                try:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                        # Print the exact batch dictionary we're passing to the model
                        logging.debug("Passing the following keys to model: " + str(list(batch_dict.keys())))
                        
                        # Additional debugging for audio/video shapes
                        if "audio" in batch_dict and batch_dict["audio"] is not None:
                            logging.debug(f"Audio tensor shape: {batch_dict['audio'].shape}, dtype: {batch_dict['audio'].dtype}")
                        if "video" in batch_dict and batch_dict["video"] is not None:
                            logging.debug(f"Video tensor shape: {batch_dict['video'].shape}, dtype: {batch_dict['video'].dtype}")
                        if "labels" in batch_dict and batch_dict["labels"] is not None:
                            if isinstance(batch_dict["labels"], torch.Tensor):
                                logging.debug(f"Labels tensor shape: {batch_dict['labels'].shape}, dtype: {batch_dict['labels'].dtype}")
                            else:
                                logging.debug(f"Labels type: {type(batch_dict['labels'])}")
                        
                        # Try running the model
                        outputs = model(**batch_dict)
                        
                        if isinstance(outputs, dict) and "loss" in outputs:
                            loss = outputs["loss"]
                            # Real loss was calculated - this is good!
                            logging.info(f"Successfully calculated loss: {loss}")
                        else:
                            logging.warning(f"No loss in model outputs. Keys: {outputs.keys() if isinstance(outputs, dict) else 'not a dict'}")
                            # If no loss, set a dummy loss to continue
                            loss = torch.tensor(1e6, device=device)
                    
                except TypeError as e:
                    # If we get a TypeError, it's likely because we're passing wrong args
                    logging.error(f"Type error in model forward pass: {e}")
                    
                    # Check if this is a weights issue by examining the error message
                    error_str = str(e)
                    if "numpy array is not a tensor" in error_str or "weight" in error_str or "Missing key" in error_str:
                        logging.error("This appears to be a model weight initialization issue!")
                        logging.error("The model may not have loaded pre-trained weights correctly.")
                    
                    # Try to inspect the model's forward method to see what it expects
                    try:
                        import inspect
                        forward_sig = inspect.signature(model.forward)
                        logging.error(f"Model's forward method expects: {forward_sig}")
                        
                        # Print the full model class for reference
                        logging.debug(f"Model class name: {model.__class__.__name__}")
                        if hasattr(model, 'config'):
                            logging.debug(f"Model config: {model.config}")
                            
                        # Try to get more information about the model state
                        for name, module in model.named_children():
                            try:
                                param_count = sum(p.numel() for p in module.parameters())
                                requires_grad = any(p.requires_grad for p in module.parameters())
                                logging.debug(f"Module '{name}': {param_count:,} params, requires_grad={requires_grad}")
                            except Exception:
                                logging.debug(f"Module '{name}': Could not count parameters")
                    except Exception:
                        pass
                    
                    # Create a dummy loss to continue with validation
                    loss = torch.tensor(1e6, device=device)
                    
                except Exception as e:
                    logging.error(f"Other error in model forward pass: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Create a dummy loss to continue with validation
                    loss = torch.tensor(1e6, device=device)
                
                # Add stability checks
                if isinstance(loss, torch.Tensor) and (torch.isnan(loss) or torch.isinf(loss)):
                    logging.warning(f"Found unstable loss value: {loss}. Setting to large finite value")
                    loss = torch.tensor(1e6, device=loss.device, dtype=loss.dtype)
                    
                # Update progress bar
                loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
                pbar.set_description(f"Validation | Loss: {loss_value:.6f}")
                
                # Update statistics
                if "audio" in batch and batch["audio"] is not None:
                    batch_size = batch["audio"].size(0)
                elif "video" in batch and batch["video"] is not None:
                    batch_size = batch["video"].size(0)
                else:
                    batch_size = 1
                    
                total_loss += loss_value * batch_size
                total_samples += batch_size
                
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate average loss
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    
    # Log validation results
    logging.info(f"Validation complete | Avg Loss: {avg_loss:.6f}")
    
    return avg_loss

def process_batch(batch, model, device):
    """Process a batch for the model"""
    if batch is None:
        return {"loss": torch.tensor(0.0, device=device)}
    
    # Validate input shapes
    batch_size = None
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if batch_size is None:
                batch_size = v.size(0)
            elif v.size(0) != batch_size:
                logging.error(f"Batch size mismatch in {k}: {v.size(0)} vs {batch_size}")
                # Handle mismatch by truncating or padding
                if v.size(0) > batch_size:
                    batch[k] = v[:batch_size]
                else:
                    padding = torch.zeros((batch_size - v.size(0), *v.shape[1:]), 
                                       dtype=v.dtype, device=v.device)
                    batch[k] = torch.cat([v, padding], dim=0)
    
    # Verify model components are properly initialized
    if not hasattr(model, 'whisper_model') or model.whisper_model is None:
        logging.error("Whisper model not properly initialized")
        return {"loss": torch.tensor(1e6, device=device)}
    
    if not hasattr(model, 'clip_model') or model.clip_model is None:
        logging.error("CLIP model not properly initialized")
        return {"loss": torch.tensor(1e6, device=device)}
    
    if not hasattr(model, 'llm') or model.llm is None:
        logging.error("LLM not properly initialized")
        return {"loss": torch.tensor(1e6, device=device)}
    
    # Process batch
    batch_dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dict[k] = v.to(device)
        elif k == "labels" and isinstance(v, list):
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                try:
                    encoded = model.tokenizer(v, return_tensors="pt", padding=True, truncation=True)
                    batch_dict[k] = encoded.input_ids.to(device)
                except Exception as e:
                    logging.error(f"Failed to tokenize labels: {e}")
                    return {"loss": torch.tensor(1e6, device=device)}
            else:
                logging.error("Tokenizer not available for label processing")
                return {"loss": torch.tensor(1e6, device=device)}
        else:
            batch_dict[k] = v
    
    # Verify final batch shapes
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor):
            logging.debug(f"Final {k} shape: {v.shape}")
            if v.size(0) != batch_size:
                logging.error(f"Final batch size mismatch in {k}: {v.size(0)} vs {batch_size}")
                return {"loss": torch.tensor(1e6, device=device)}
    
    batch_dict["return_loss"] = True
    return batch_dict

def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()]
    )
    
    # Load config
    config = load_config(args.config)
    
    # Override config with args
    if args.modality:
        config.model.modality = args.modality
    
    # Create output dir for validation
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logging.info("Loading model (this might take a minute)...")
    try:
        # Create model instance
        model = ClipWhisperModel(
            llm_path=config.model.llm_path,
            whisper_model=config.model.whisper_model,
            clip_model=config.model.clip_model,
            modality=config.model.modality,
            use_lora=config.model.use_lora,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            max_seq_len=config.data.max_seq_len,
            device=device
        )
        
        # Load checkpoint if provided
        if args.checkpoint_path:
            logging.info(f"Loading model checkpoint from {args.checkpoint_path}")
            try:
                checkpoint = torch.load(args.checkpoint_path, map_location=device)
                
                # Check if this is a trainer state dict or just model weights
                if "model" in checkpoint:
                    # This is a trainer state dict
                    model_state = checkpoint["model"]
                    logging.info(f"Loaded checkpoint from trainer state (keys: {list(checkpoint.keys())})")
                else:
                    # This is just model weights
                    model_state = checkpoint
                    logging.info("Loaded raw model weights")
                
                # Try to load the state dict
                missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                
                if missing_keys:
                    logging.warning(f"Missing keys when loading checkpoint: {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
                    
                logging.info("✓ Checkpoint loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            logging.error("No checkpoint provided. Validation requires a trained checkpoint.")
            return 1
            
        model.to(device)
        
        # Check model weights were loaded
        logging.info(f"Model created successfully. Checking components...")
        
        # Check that key components exist
        if hasattr(model, 'whisper_model'):
            logging.info(f"✓ Whisper model loaded: {type(model.whisper_model).__name__}")
            # Verify Whisper has parameters
            whisper_params = sum(p.numel() for p in model.whisper_model.parameters())
            logging.info(f"  - Whisper parameters: {whisper_params:,}")
        else:
            logging.error("✗ Whisper model not found!")
            
        if hasattr(model, 'clip_model'):
            logging.info(f"✓ CLIP model loaded: {type(model.clip_model).__name__}")
            # Verify CLIP has parameters
            clip_params = sum(p.numel() for p in model.clip_model.parameters())
            logging.info(f"  - CLIP parameters: {clip_params:,}")
        else:
            logging.error("✗ CLIP model not found!")
            
        if hasattr(model, 'llm'):
            logging.info(f"✓ LLM loaded: {type(model.llm).__name__}")
            # Verify LLM has parameters
            llm_params = sum(p.numel() for p in model.llm.parameters())
            logging.info(f"  - LLM parameters: {llm_params:,}")
        else:
            logging.error("✗ LLM not found!")
            
        # Check connector exists (if using both modalities)
        if config.model.modality == "both" and hasattr(model, 'connector'):
            logging.info(f"✓ Connector loaded: {type(model.connector).__name__}")
        elif config.model.modality == "both":
            logging.error("✗ Connector not found but modality is 'both'!")
            
        # Count total trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Total parameters: {total_params:,} (Trainable: {trainable_params:,})")
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create validation dataloader with batch size override if provided
    val_dataloader, tokenizer = create_validation_loader(
        config, 
        args.data_path, 
        args.subset_size,
        args.batch_size
    )
    
    if val_dataloader is None:
        logging.error("Failed to create validation dataloader")
        return 1
    
    # Run validation
    logging.info(f"Running quick validation with {args.subset_size} samples...")
    try:
        # Run validation directly
        val_loss = simple_validate(model, val_dataloader, device)
        
        logging.info(f"Validation complete. Loss: {val_loss}")
        
        # Check if we got the dummy loss (1e6) which indicates a failed run but caught error
        if abs(val_loss - 1e6) < 1.0:
            logging.error("Validation produced only dummy loss values! The model did not run successfully.")
            return 1
        
        if val_loss == float('inf') or (isinstance(val_loss, torch.Tensor) and torch.isnan(val_loss)):
            logging.error("Validation produced infinite or NaN loss!")
            return 1
            
        return 0
    except Exception as e:
        logging.error(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 