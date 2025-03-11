#!/usr/bin/env python3
"""
Decode script for the ClipWhisperModel that supports WER calculation.
This script loads a trained model and runs inference on test data.
"""

import os
import sys
import argparse
import logging
import torch
import datetime
import yaml
import traceback
from tqdm import tqdm
import jiwer
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.clip_whisper.models import ClipWhisperModel
from src.clip_whisper.data.simple_dataset import AVSRDataset
from transformers import WhisperProcessor, CLIPProcessor, AutoTokenizer, WhisperModel, CLIPVisionModel, AutoModelForCausalLM
from src.utils.media import load_audio, load_video, save_results
from src.clip_whisper.models import ModalityConnector

def calculate_wer(references, hypotheses):
    """Calculate Word Error Rate using jiwer"""
    try:
        wer = jiwer.wer(references, hypotheses)
        return wer
    except Exception as e:
        logging.error(f"Error calculating WER: {e}")
        return float('inf')

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run inference with the ClipWhisperModel and calculate WER")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--test_data", type=str, help="Path to test data TSV file")
    parser.add_argument("--test_wrd", type=str, help="Path to test word reference file")
    parser.add_argument("--output_dir", type=str, default="outputs/clip_whisper_decoding", help="Directory to save decoding results")
    parser.add_argument("--modality", type=str, default="both", choices=["audio", "video", "both"], 
                      help="Modality to use for inference")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="Device to run inference on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/clip_whisper.yaml", 
                      help="Configuration file for processor settings")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--whisper_model", type=str, default="checkpoints/whisper-medium",
                      help="Path to pre-trained Whisper model")
    parser.add_argument("--clip_model", type=str, default="checkpoints/clip-vit-base-patch32",
                      help="Path to pre-trained CLIP model")
    parser.add_argument("--llm_model", type=str, default="checkpoints/Llama-3.2-1B",
                      help="Path to pre-trained LLM model")
    parser.add_argument('--calculate_loss', action='store_true', 
                        help='Calculate loss during decoding (requires reference text)')
    parser.add_argument('--text_key', type=str, default='text', 
                        help='Key for text input in batch (default: "text")')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"decode_{timestamp}.log")
    results_file = os.path.join(args.output_dir, f"results_{timestamp}.txt")
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log decoding parameters
    logging.info("=" * 80)
    logging.info("CLIP-WHISPER DECODING")
    logging.info("=" * 80)
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Modality: {args.modality}")
    logging.info(f"Test data: {args.test_data}")
    logging.info(f"Test references: {args.test_wrd}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Max new tokens: {args.max_new_tokens}")
    logging.info("=" * 80)
    
    try:
        # Load configuration
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Load model
        logging.info(f"Loading model from {args.model_path}")
        try:
            # Check if the necessary files exist
            audio_connector_path = os.path.join(args.model_path, "audio_connector.pt")
            video_connector_path = os.path.join(args.model_path, "video_connector.pt")
            model_config_path = os.path.join(args.model_path, "model_config.json")
            tokenizer_path = os.path.join(args.model_path, "tokenizer")
            llm_path = os.path.join(args.model_path, "llm")
            
            if not os.path.exists(tokenizer_path):
                logging.error(f"Tokenizer directory not found at {tokenizer_path}")
                return
            
            if not os.path.exists(model_config_path):
                logging.error(f"Model config not found at {model_config_path}")
                return
                
            if not os.path.exists(video_connector_path):
                logging.error(f"Video connector not found at {video_connector_path}")
                return
                
            # Load model configuration to get modality
            with open(model_config_path, "rb") as f:
                model_config = torch.load(f)
                
            # Load the config.json to get model settings
            config_path = os.path.join(args.model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                config_data = {}
            
            # Extract settings
            whisper_model_name = config_data.get("whisper_model", "openai/whisper-medium")
            clip_model_name = config_data.get("clip_model", "openai/clip-vit-base-patch32")
            llm_model_name = config_data.get("llm_path", "meta-llama/Llama-2-7b-chat-hf")
            
            # Load tokenizer
            logging.info(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # Load models from provided paths or HuggingFace
            if os.path.exists(args.whisper_model):
                logging.info(f"Loading Whisper model from local path: {args.whisper_model}")
                whisper_model = WhisperModel.from_pretrained(args.whisper_model)
                whisper_processor = WhisperProcessor.from_pretrained(args.whisper_model)
            else:
                logging.info(f"Loading Whisper model from HuggingFace: {whisper_model_name}")
                whisper_model = WhisperModel.from_pretrained(whisper_model_name)
                whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
            
            if os.path.exists(args.clip_model):
                logging.info(f"Loading CLIP model from local path: {args.clip_model}")
                clip_model = CLIPVisionModel.from_pretrained(args.clip_model)
                clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
            else:
                logging.info(f"Loading CLIP model from HuggingFace: {clip_model_name}")
                clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
                clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            
            # Load LLM - use the provided path instead of the one in the model directory
            if os.path.exists(args.llm_model):
                logging.info(f"Loading LLM from local path: {args.llm_model}")
                llm = AutoModelForCausalLM.from_pretrained(args.llm_model, device_map="auto")
            else:
                # Fall back to the trained model's adapter if the specified model doesn't exist
                logging.info(f"LLM path {args.llm_model} doesn't exist, falling back to HuggingFace: {llm_model_name}")
                llm = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto")
            
            # Create a ClipWhisperModel with pre-loaded components
            model = ClipWhisperModel(
                llm_path=args.llm_model,  # Pass for reference
                whisper_model=args.whisper_model,  # Pass for reference
                clip_model=args.clip_model,  # Pass for reference
                device=args.device,
                use_lora=False,  # LoRA is already applied in the saved model
                modality=args.modality,
                _provided_tokenizer=tokenizer  # Use our pre-loaded tokenizer
            )
            
            # Manually assign the pre-loaded components to the model to avoid loading them again
            model.whisper = whisper_model
            model.whisper_processor = whisper_processor
            model.clip = clip_model
            model.clip_processor = clip_processor
            model.llm = llm
            model.tokenizer = tokenizer
            
            # Set dimensions
            model.audio_dim = whisper_model.config.d_model
            model.video_dim = clip_model.config.hidden_size
            
            # Determine LLM dimension
            model.llm_dim = model._get_llm_dim()
            
            # Load the connectors
            model.audio_connector = ModalityConnector(
                input_dim=model.audio_dim,
                output_dim=model.llm_dim,
                device=args.device
            )
            model.audio_connector.load_state_dict(torch.load(audio_connector_path))
            
            model.video_connector = ModalityConnector(
                input_dim=model.video_dim,
                output_dim=model.llm_dim,
                device=args.device
            )
            model.video_connector.load_state_dict(torch.load(video_connector_path))
            
            # Set other parameters from model_config
            model.max_seq_len = model_config.get("max_seq_len", 256)
            model.fusion_scale = model_config.get("fusion_scale", 0.5)
            
            # Override with command line modality
            model.modality = args.modality
            logging.info(f"Using modality: {args.modality}")
            print(f"Using modality: {args.modality}")
            
            # Move model to device
            model.to(args.device)
            model.eval()
            
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error during model loading: {e}")
            logging.error(traceback.format_exc())
            return
        
        # Batch decoding mode requires test data and reference file
        if not args.test_data or not args.test_wrd:
            logging.error("Both --test_data and --test_wrd are required for batch decoding mode")
            print("ERROR: Both --test_data and --test_wrd are required for batch decoding mode")
            return
        
        # Load test data
        logging.info(f"Loading test data from {args.test_data}")
        
        # Read test.wrd for reference texts
        references = {}
        try:
            # First load utterance IDs from test.tsv to understand their format
            utterance_ids = []
            with open(args.test_data, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        # Extract the utterance ID from the first column
                        utt_id = parts[0]
                        # Skip header line if present
                        if utt_id != "/" and not utt_id.startswith("#"):
                            utterance_ids.append(utt_id)
            
            logging.info(f"Found {len(utterance_ids)} utterance IDs in test data")
            
            # Now load reference texts
            with open(args.test_wrd, 'r') as f:
                lines = f.readlines()
                
                # Debug information about reference file
                logging.info(f"Reference file contains {len(lines)} lines")
                if len(lines) > 0:
                    logging.info(f"First line of reference file: '{lines[0].strip()}'")
                if len(lines) > 1:
                    logging.info(f"Second line of reference file: '{lines[1].strip()}'")

                # Check if we have utterance IDs from test.tsv in a specific format
                # Extract all parts of the utterance IDs for flexible matching
                utterance_id_parts = {}
                for utt_id in utterance_ids:
                    # Store the full ID
                    utterance_id_parts[utt_id] = utt_id
                    
                    # Store without path prefix
                    if "/" in utt_id:
                        simple_id = utt_id.split("/")[-1]
                        utterance_id_parts[simple_id] = utt_id
                    
                    # Store numeric part if it exists
                    if any(char.isdigit() for char in utt_id):
                        numeric_parts = ''.join(char for char in utt_id if char.isdigit())
                        if numeric_parts:
                            utterance_id_parts[numeric_parts] = utt_id
                
                # Detect reference file format based on first few lines
                has_id_prefix = False
                if len(lines) > 0:
                    parts = lines[0].strip().split(maxsplit=1)
                    if len(parts) == 2 and (parts[0] in utterance_ids or parts[0] in utterance_id_parts):
                        has_id_prefix = True
                        logging.info("Detected reference format with ID prefix")
                    else:
                        logging.info("Detected reference format without ID prefix, assuming line-by-line format")
                
                # If number of lines in WRD matches number of utterances,
                # and no ID prefix is detected, assume each line corresponds to one utterance
                if len(lines) == len(utterance_ids) and not has_id_prefix:
                    logging.info("Number of reference lines matches number of utterance IDs, using direct mapping")
                    for i, line in enumerate(lines):
                        if line.strip():
                            references[utterance_ids[i]] = line.strip()
                            
                            # Also add simplified versions of the ID to the references dict
                            if "/" in utterance_ids[i]:
                                simple_id = utterance_ids[i].split("/")[-1]
                                references[simple_id] = line.strip()
                else:
                    # Otherwise, try to parse as ID + text format
                    logging.info("Parsing references with ID prefix format")
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2:
                                utt_id = parts[0]
                                ref_text = parts[1]
                                references[utt_id] = ref_text
                                
                                # If the ID is in our utterance_id_parts mapping, add references
                                # for all forms of the ID
                                if utt_id in utterance_id_parts:
                                    full_id = utterance_id_parts[utt_id]
                                    references[full_id] = ref_text
                                    
                                    # Also add simplified version
                                    if "/" in full_id:
                                        simple_id = full_id.split("/")[-1]
                                        references[simple_id] = ref_text
                            elif len(parts) == 1:
                                # Handle case with just an ID and no text
                                references[parts[0]] = ""
                
                # Log some statistics about loaded references
                logging.info(f"Loaded {len(references)} reference transcriptions")
                logging.info(f"First few reference IDs: {list(references.keys())[:5]}")
                logging.info(f"Number of utterance IDs matching references: {sum(1 for uid in utterance_ids if uid in references)}")
                
                # Check if we have enough matching IDs
                matching_percent = sum(1 for uid in utterance_ids if uid in references) / len(utterance_ids) * 100
                if matching_percent < 50:
                    logging.warning(f"Only {matching_percent:.1f}% of utterance IDs match references. Check ID formats!")
                    
                    # Try to determine if we need to transform IDs
                    logging.info("Example utterance ID formats:")
                    for uid in utterance_ids[:5]:
                        logging.info(f"  {uid}")
                    
                    logging.info("Example reference ID formats:")
                    for rid in list(references.keys())[:5]:
                        logging.info(f"  {rid}")
        except Exception as e:
            logging.error(f"Error loading references: {e}")
            logging.error(traceback.format_exc())
        
        # Get root directory from TSV file path (common parent directory)
        root_dir = os.path.dirname(os.path.dirname(args.test_data))
        
        # Read the test.wrd file
        label_path = args.test_wrd
        if not os.path.exists(label_path):
            logging.error(f"Reference file not found: {label_path}")
            return
            
        # Create test dataset
        try:
            test_dataset = AVSRDataset(
                manifest_path=args.test_data,
                label_path=label_path,
                root_dir=root_dir,
                whisper_processor=model.whisper_processor,
                clip_processor=model.clip_processor,
                tokenizer=model.tokenizer,
                max_audio_length=30,  # seconds
                max_video_length=300,  # frames
                split="test",
            )
            
            # Create dataloader
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=AVSRDataset.collate_fn,  # Use the class method
            )
            
            logging.info(f"Created dataloader with {len(test_dataset)} samples")
        except Exception as e:
            logging.error(f"Error creating dataset/dataloader: {e}")
            logging.error(traceback.format_exc())
            return
        
        # Initialize results
        all_references = []
        all_hypotheses = []
        results = []
        
        # Decode each batch
        logging.info(f"Starting decoding using modality: {args.modality}...")
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Decoding")):
            if batch is None:
                logging.warning(f"Batch {batch_idx} is None, skipping")
                continue
                
            try:
                # Get utterance IDs
                if "utt_id" in batch:
                    utt_ids = batch["utt_id"]
                    if args.verbose:
                        logging.info(f"Found utterance IDs in batch: {utt_ids}")
                else:
                    # If batch doesn't contain utt_ids, create artificial ones based on batch index
                    # Use a reasonable default since batch_hypotheses isn't created yet
                    utt_ids = [f"sample_{batch_idx}_{i}" for i in range(1)]  # We'll extend this later if needed
                    logging.warning(f"No utterance IDs found in batch {batch_idx}, using auto-generated IDs")
                
                # Print the actual keys in the batch for debugging
                if args.verbose and batch_idx < 2:  # Only print for first few batches to avoid spam
                    logging.info(f"Batch {batch_idx} keys: {list(batch.keys())}")
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            logging.info(f"  {key}: {batch[key].shape}")
                        else:
                            logging.info(f"  {key}: {type(batch[key])}")
                
                # Prepare inputs based on modality
                inputs_for_model = {}
                
                # Add audio input if available
                if "audio" in batch and args.modality in ["audio", "both"]:
                    inputs_for_model["audio"] = batch["audio"].to(args.device)

                # Add video input if available
                if "video" in batch and args.modality in ["video", "both"]:
                    inputs_for_model["video"] = batch["video"].to(args.device)

                # Add text input if available (use 'labels' or 'text' as in training)
                if "labels" in batch:
                    inputs_for_model["labels"] = batch["labels"].to(args.device)
                elif "text" in batch:
                    inputs_for_model["text"] = batch["text"].to(args.device)
                
                # Special check for video format seen in the logs
                if args.verbose and batch_idx < 5:
                    logging.info(f"Raw batch type: {type(batch)}")
                    if hasattr(batch, 'keys'):
                        for k in batch.keys():
                            if isinstance(batch[k], torch.Tensor):
                                logging.info(f"  Key: {k}, Shape: {batch[k].shape}, Type: {batch[k].dtype}")
                            else:
                                logging.info(f"  Key: {k}, Type: {type(batch[k])}")
                
                # Special case: if we have a CLIP processor output, pixel_values might be nested in a dict
                if "video_input" not in inputs_for_model and args.modality in ["video", "both"]:
                    # Look for nested pixel_values in a dictionary
                    for key, value in batch.items():
                        if isinstance(value, dict) and "pixel_values" in value:
                            inputs_for_model["video_input"] = value["pixel_values"].to(args.device)
                            if args.verbose:
                                logging.info(f"Found nested video features in key: {key}")
                            break
                
                # Based on the logs showing "Processed video shape: torch.Size([42, 3, 224, 224])"
                if "video_input" not in inputs_for_model and args.modality in ["video", "both"] and 0 in batch:
                    video_tensor = batch[0]
                    if isinstance(video_tensor, torch.Tensor) and len(video_tensor.shape) == 4:
                        # Format like [frames, channels, height, width]
                        if args.verbose:
                            logging.info(f"Found video at batch[0], shape: {video_tensor.shape}")
                        video_tensor = video_tensor.unsqueeze(0)  # Add batch dim [1, frames, channels, height, width]
                        inputs_for_model["video_input"] = video_tensor.to(args.device)
                
                # Try even more approaches to find video data if we still don't have it
                if "video_input" not in inputs_for_model and args.modality in ["video", "both"]:
                    # Last resort: Look for any tensor that could be a video
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) >= 4:
                            # This looks like a video tensor with shape [batch, frames, channels, height, width]
                            # or [frames, channels, height, width]
                            if args.verbose:
                                logging.info(f"Found tensor that looks like video in key: {key}, shape: {value.shape}")
                            
                            # Ensure it has the right dimensions for a video
                            if len(value.shape) == 4:  # [frames, channels, height, width]
                                # Add batch dimension
                                value = value.unsqueeze(0)
                            
                            inputs_for_model["video_input"] = value.to(args.device)
                            break
                
                # Final fallback: If batch is directly a tensor, use it
                if "video_input" not in inputs_for_model and args.modality in ["video", "both"]:
                    if isinstance(batch, torch.Tensor) and len(batch.shape) >= 4:
                        if args.verbose:
                            logging.info(f"Using batch directly as video input, shape: {batch.shape}")
                        inputs_for_model["video_input"] = batch.to(args.device)
                
                # Ensure we have the right inputs for the selected modality
                if args.modality == "audio" and "audio_input" not in inputs_for_model:
                    logging.warning(f"Batch {batch_idx}: Audio modality selected but no audio input in batch, skipping")
                    continue
                elif args.modality == "video" and "video_input" not in inputs_for_model:
                    logging.warning(f"Batch {batch_idx}: Video modality selected but no video input in batch, skipping")
                    continue
                elif args.modality == "both" and "audio_input" not in inputs_for_model and "video_input" not in inputs_for_model:
                    logging.warning(f"Batch {batch_idx}: Both modality selected but neither audio nor video input in batch, skipping")
                    continue
                
                # Run model forward pass
                with torch.no_grad():
                    outputs = model(**inputs_for_model, return_loss=True)

                # Calculate loss if text/labels are available
                if "labels" in inputs_for_model or "text" in inputs_for_model:
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        labels = inputs_for_model.get("labels", inputs_for_model.get("text"))
                        loss = model.calculate_loss(logits, labels)
                        logging.info(f"Batch {batch_idx} - Loss: {loss.item():.4f}")
                
                # Process outputs
                batch_hypotheses = []
                if isinstance(outputs, list) and all(isinstance(item, str) for item in outputs):
                    # If outputs is already a list of strings
                    batch_hypotheses = outputs
                else:
                    # If outputs is token IDs, decode them
                    try:
                        if isinstance(outputs, torch.Tensor):
                            # Single tensor output
                            text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            batch_hypotheses.append(text)
                        elif hasattr(outputs, "__getitem__"):
                            # Sequence output
                            for output_ids in outputs:
                                if isinstance(output_ids, torch.Tensor):
                                    text = model.tokenizer.decode(output_ids, skip_special_tokens=True)
                                else:
                                    # Handle case where output might already be text
                                    text = str(output_ids)
                                batch_hypotheses.append(text)
                        else:
                            # Fallback for other output types
                            text = str(outputs)
                            batch_hypotheses.append(text)
                    except Exception as e:
                        logging.error(f"Error processing model outputs: {e}")
                        batch_hypotheses = ["[ERROR: Failed to decode output]"]
                
                # Decode the output tokens for each sample
                batch_size = len(batch_hypotheses)
                
                # If we need more utterance IDs (e.g., if the model output more hypotheses than inputs)
                if len(utt_ids) < batch_size:
                    # Extend utt_ids with auto-generated IDs
                    for i in range(len(utt_ids), batch_size):
                        utt_ids.append(f"sample_{batch_idx}_{i}")
                    logging.warning(f"Extended utterance IDs with auto-generated IDs for batch {batch_idx}")
                
                for i in range(batch_size):
                    # Get actual utterance ID from dataset if available
                    utt_id = utt_ids[i] if i < len(utt_ids) else f"sample_{batch_idx}_{i}"
                    
                    # For debug purposes, if it's an auto-generated ID, log this
                    if utt_id.startswith("sample_"):
                        logging.warning(f"Using auto-generated utterance ID: {utt_id}")
                    
                    hypothesis = batch_hypotheses[i]
                    
                    # Map the utterance ID to the format in the reference file if needed
                    # Check if utterance ID exists directly in references
                    reference = references.get(utt_id, "")
                    
                    # Print the transcription to console for immediate feedback
                    print(f"\n=== Transcription for {utt_id} ===")
                    print(f"HYP: {hypothesis}")
                    if reference:
                        print(f"REF: {reference}")
                    else:
                        # Try to find reference by removing the path prefix if present
                        if "/" in utt_id:
                            # Extract just the last part of the path as a possible ID
                            simple_id = utt_id.split("/")[-1]
                            reference = references.get(simple_id, "")
                            if reference:
                                print(f"REF: {reference}")
                                print(f"(Found reference using simplified ID: {simple_id})")
                                # Update utt_id to the simplified version for WER calculation
                                utt_id = simple_id
                        
                    # Calculate WER for this sample
                    if reference:
                        sample_wer = calculate_wer([reference], [hypothesis])
                        all_references.append(reference)
                        all_hypotheses.append(hypothesis)
                    else:
                        sample_wer = float('inf')
                        # Enhanced error reporting for missing references
                        print(f"No reference found for utterance ID: {utt_id}")
                        print(f"Available reference IDs: {list(references.keys())[:5]}... (showing first 5 of {len(references)})")
                        if utt_id.startswith("sample_"):
                            print("Note: Using auto-generated sample IDs. These might not match reference file.")
                            print("Check format of reference file and ensure utterance IDs match between data and references.")
                        logging.warning(f"No reference found for utterance {utt_id}")
                    
                    # Save result
                    results.append({
                        "utt_id": utt_id,
                        "reference": reference,
                        "hypothesis": hypothesis,
                        "wer": sample_wer
                    })
                    
                    # Log individual results if verbose
                    if args.verbose:
                        logging.info(f"Utterance: {utt_id}")
                        logging.info(f"  REF: {reference}")
                        logging.info(f"  HYP: {hypothesis}")
                        logging.info(f"  WER: {sample_wer:.4f}")
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                logging.error(traceback.format_exc())
                continue
        
        # Save WER results
        if len(all_references) > 0:
            overall_wer = calculate_wer(all_references, all_hypotheses)
            
            logging.info(f"Overall WER: {overall_wer:.4f}")
            
            # Save detailed results
            with open(results_file, 'w') as f:
                f.write(f"Modality: {args.modality}\n")
                f.write(f"Overall WER: {overall_wer:.4f}\n\n")
                f.write(f"Detailed Results:\n")
                f.write(f"{'Utterance ID':<20} {'WER':<10} {'Reference':<40} {'Hypothesis':<40}\n")
                f.write("-" * 110 + "\n")
                
                for result in results:
                    f.write(f"{result['utt_id']:<20} {result['wer']:<10.4f} {result['reference'][:40]:<40} {result['hypothesis'][:40]:<40}\n")
                    
            logging.info(f"Results saved to {results_file}")
            
            # Also save summary to a separate file
            with open(os.path.join(args.output_dir, f"wer_{timestamp}.txt"), "w") as f:
                f.write(f"Overall WER: {overall_wer:.4f}\n")
                f.write(f"Total samples: {len(all_references)}\n")
            
            # Print summary to console
            print("\n" + "="*80)
            print(f"DECODING SUMMARY")
            print(f"Overall WER: {overall_wer:.4f}")
            print(f"Total samples successfully decoded: {len(all_references)}")
            print("="*80)
        else:
            logging.warning("No samples were successfully processed for WER calculation")
    
    except Exception as e:
        logging.error(f"Error during decoding: {e}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    main() 