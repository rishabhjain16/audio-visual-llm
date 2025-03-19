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
    parser.add_argument("--modality", type=str, choices=["audio", "video", "both"], default="both", help="Which modality to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for decoding")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation (1.0=neutral, <1.0=more focused, >1.0=more random)")
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
    parser.add_argument("--output_file", type=str, default="decode_results.json", help="Output file for results")
    
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
    
    # Configure console output to be minimal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    
    # Set up a clean file handler for detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Replace the default handlers with our custom ones
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Print header for console output
    print("\n" + "=" * 80)
    print("CLIP-WHISPER DECODING")
    print(f"Model: {args.model_path}")
    print(f"Modality: {args.modality}")
    print("=" * 80 + "\n")
    
    try:
        # Load configuration
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Load model
        logging.info(f"Loading model from {args.model_path}")
        try:
            # First, load pre-trained models and their tokenizers
            # Load tokenizer from LLM model path
            logging.info(f"Loading tokenizer from {args.llm_model}")
            tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
            
            # Load models from provided paths or HuggingFace
            if os.path.exists(args.whisper_model):
                logging.info(f"Loading Whisper model from local path: {args.whisper_model}")
                whisper_model = WhisperModel.from_pretrained(args.whisper_model)
                whisper_processor = WhisperProcessor.from_pretrained(args.whisper_model)
            else:
                whisper_model_name = "openai/whisper-medium"
                logging.info(f"Loading Whisper model from HuggingFace: {whisper_model_name}")
                whisper_model = WhisperModel.from_pretrained(whisper_model_name)
                whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
            
            if os.path.exists(args.clip_model):
                logging.info(f"Loading CLIP model from local path: {args.clip_model}")
                clip_model = CLIPVisionModel.from_pretrained(args.clip_model)
                clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
            else:
                clip_model_name = "openai/clip-vit-base-patch32"
                logging.info(f"Loading CLIP model from HuggingFace: {clip_model_name}")
                clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
                clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            
            # Load LLM from provided path with 4-bit quantization to save memory
            logging.info(f"Loading LLM with 4-bit quantization from: {args.llm_model}")
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                llm = AutoModelForCausalLM.from_pretrained(
                    args.llm_model,
                    device_map="auto",
                    quantization_config=quantization_config
                )
                logging.info("Successfully loaded LLM with 4-bit quantization")
            except ImportError:
                logging.warning("BitsAndBytes not available, loading LLM in standard mode")
                llm = AutoModelForCausalLM.from_pretrained(
                    args.llm_model,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            except Exception as e:
                logging.warning(f"Failed to load with 4-bit quantization: {e}")
                logging.warning("Falling back to standard loading with float16")
                llm = AutoModelForCausalLM.from_pretrained(
                    args.llm_model,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            
            # Create a ClipWhisperModel with our pre-loaded components
            logging.info("Creating ClipWhisperModel with pre-loaded components")
            model = ClipWhisperModel(
                llm_path=None,  # Don't load LLM again
                whisper_model=None,  # Don't load Whisper again
                clip_model=None,  # Don't load CLIP again
                device=args.device,
                use_lora=False,
                modality=args.modality,
                _provided_tokenizer=tokenizer,
                _provided_llm=llm,  # Pass already loaded LLM
                _provided_whisper=whisper_model,  # Pass already loaded Whisper
                _provided_clip=clip_model  # Pass already loaded CLIP
            )
            
            # Manually assign processors that weren't passed in the constructor
            model.whisper_processor = whisper_processor
            model.clip_processor = clip_processor
            
            # Set dimensions
            model.audio_dim = whisper_model.config.d_model
            model.video_dim = clip_model.config.hidden_size
            
            # Determine LLM dimension
            model.llm_dim = model._get_llm_dim()
            
            # Initialize connectors
            model.audio_connector = ModalityConnector(
                input_dim=model.audio_dim,
                output_dim=model.llm_dim,
                device=args.device
            )
            
            model.video_connector = ModalityConnector(
                input_dim=model.video_dim,
                output_dim=model.llm_dim,
                device=args.device
            )
            
            # Load the weights from the single checkpoint file
            logging.info(f"Loading model weights from checkpoint: {args.model_path}")
            if os.path.isfile(args.model_path):
                # Load checkpoint on CPU first to avoid CUDA OOM
                logging.info("Loading checkpoint on CPU first to avoid memory issues...")
                checkpoint = torch.load(args.model_path, map_location='cpu')
                
                # Check if it's a direct state_dict or has a 'model_state_dict' key
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Extract only the connector weights to save memory
                audio_connector_dict = {k.replace('audio_connector.', ''): v for k, v in state_dict.items() if 'audio_connector' in k}
                video_connector_dict = {k.replace('video_connector.', ''): v for k, v in state_dict.items() if 'video_connector' in k}
                
                # Clear the full state_dict to free memory
                del state_dict
                del checkpoint
                torch.cuda.empty_cache()  # Explicitly free GPU memory
                
                # Load connector weights
                if audio_connector_dict:
                    # Move weights to the target device just before loading
                    audio_connector_dict = {k: v.to(args.device) for k, v in audio_connector_dict.items()}
                    model.audio_connector.load_state_dict(audio_connector_dict)
                    logging.info("Loaded audio connector weights from checkpoint")
                else:
                    logging.warning("No audio connector weights found in checkpoint")
                
                if video_connector_dict:
                    # Move weights to the target device just before loading
                    video_connector_dict = {k: v.to(args.device) for k, v in video_connector_dict.items()}
                    model.video_connector.load_state_dict(video_connector_dict)
                    logging.info("Loaded video connector weights from checkpoint")
                else:
                    logging.warning("No video connector weights found in checkpoint")
                
                # Try to load config separately to reduce memory usage
                try:
                    # Re-load checkpoint to get config only
                    config_only = torch.load(args.model_path, map_location='cpu')
                    if 'config' in config_only:
                        model_config = config_only['config']
                        model.max_seq_len = model_config.get("max_seq_len", 256)
                        model.fusion_scale = model_config.get("fusion_scale", 0.5)
                        logging.info(f"Loaded configuration from checkpoint: max_seq_len={model.max_seq_len}, fusion_scale={model.fusion_scale}")
                    else:
                        model.max_seq_len = 256
                        model.fusion_scale = 0.5
                        logging.info("Using default configuration: max_seq_len=256, fusion_scale=0.5")
                    
                    # Clean up
                    del config_only
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.warning(f"Failed to load config: {e}, using defaults")
                    model.max_seq_len = 256
                    model.fusion_scale = 0.5
            else:
                logging.error(f"Checkpoint file not found: {args.model_path}")
                return
            
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
        
        # Initialize results storage
        hypotheses = {}         # Store transcription outputs
        references_matched = {} # Store matched references
        all_references = []     # List of all reference texts used in WER calculation
        all_hypotheses = []     # List of all hypothesis texts used in WER calculation
        results = []            # Store detailed results by utterance
        
        # Load test data
        logging.info(f"Loading test data from {args.test_data}")
        
        # Load labels
        logging.info(f"Loading labels from {args.test_wrd}")
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
            reference_texts = []
            with open(args.test_wrd, 'r') as f:
                reference_texts = [line.strip() for line in f]
            
            # Debug information about reference file
            logging.info(f"Reference file contains {len(reference_texts)} lines")
            if len(reference_texts) > 0:
                logging.info(f"First line of reference file: '{reference_texts[0]}'")
            if len(reference_texts) > 1:
                logging.info(f"Second line of reference file: '{reference_texts[1]}'")
            
            # Match utterance IDs to reference texts - assuming they're in the same order
            if len(utterance_ids) == len(reference_texts):
                logging.info("Number of IDs matches number of references, creating 1:1 mapping")
                for i in range(len(utterance_ids)):
                    references[utterance_ids[i]] = reference_texts[i]
                    
                    # Also add simplified versions without path prefix for easier lookup
                    if "/" in utterance_ids[i]:
                        simple_id = utterance_ids[i].split("/")[-1]
                        references[simple_id] = reference_texts[i]
            else:
                logging.warning(f"Mismatch between number of utterance IDs ({len(utterance_ids)}) " 
                               f"and reference texts ({len(reference_texts)})")
                
                # If there are fewer references than IDs, use what we have
                min_len = min(len(utterance_ids), len(reference_texts))
                logging.info(f"Using first {min_len} items to create mapping")
                for i in range(min_len):
                    references[utterance_ids[i]] = reference_texts[i]
                    
                    # Also add simplified versions without path prefix
                    if "/" in utterance_ids[i]:
                        simple_id = utterance_ids[i].split("/")[-1]
                        references[simple_id] = reference_texts[i]
            
            # Log some statistics about loaded references
            logging.info(f"Loaded {len(references)} reference transcriptions")
            logging.info(f"First few reference IDs: {list(references.keys())[:5]}")
            logging.info(f"First few reference texts: {[references[k] for k in list(references.keys())[:5]]}")
            logging.info(f"Number of utterance IDs matching references: {sum(1 for uid in utterance_ids if uid in references)}")
            
            # Check if we have enough matching IDs
            matching_percent = sum(1 for uid in utterance_ids if uid in references) / len(utterance_ids) * 100
            if matching_percent < 90:
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
            
            # Get utterance IDs from the dataset
            dataset_utt_ids = []
            for i in range(len(test_dataset)):
                video_path, audio_path, audio_id = test_dataset.names[i]
                dataset_utt_ids.append(audio_id)
            
            logging.info(f"Dataset contains {len(dataset_utt_ids)} utterance IDs")
            logging.info(f"First few utterance IDs: {dataset_utt_ids[:5]}")
            
            # Check overlap with reference IDs
            matching_ids = sum(1 for uid in dataset_utt_ids if uid in references)
            matching_percent = matching_ids / len(dataset_utt_ids) * 100 if dataset_utt_ids else 0
            logging.info(f"Dataset has {matching_ids} utterance IDs that match reference IDs ({matching_percent:.1f}%)")
            
            # Create dataloader
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,  # Use shuffle=False to maintain order
                collate_fn=AVSRDataset.collate_fn,  # Use the class method
                pin_memory=False,  # Prevent CUDA OOM errors
            )
            
            logging.info(f"Created dataloader with {len(test_dataset)} samples")
        except Exception as e:
            logging.error(f"Error creating dataset/dataloader: {e}")
            logging.error(traceback.format_exc())
            return
        
        # Decode each batch
        logging.info(f"Starting decoding using modality: {args.modality}...")
        print(f"Starting decoding with modality: {args.modality}")
        print("Each hypothesis will be shown as it's generated.\n")
        
        try:
            progress_bar = tqdm(
                total=len(test_dataloader), 
                desc="Decoding", 
                unit="batch",
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
            )
            
            for batch_idx, batch in enumerate(test_dataloader):
                try:
                    logging.info(f"Processing batch {batch_idx+1}/{len(test_dataloader)}")
                    
                    # Handle different batch formats (tuple vs dictionary)
                    if isinstance(batch, tuple):
                        logging.debug(f"Batch is a tuple with {len(batch)} elements")
                        
                        # For debugging, print the shapes of the batch elements
                        for elem_idx, elem in enumerate(batch):
                            if isinstance(elem, torch.Tensor):
                                logging.debug(f"Batch element {elem_idx} is a tensor with shape: {elem.shape}")
                            elif isinstance(elem, list):
                                logging.debug(f"Batch element {elem_idx} is a list with length: {len(elem)}")
                        
                        # Get audio and video features from batch
                        audio_features = None
                        video_features = None
                        
                        # Extract audio features (first element in batch tuple)
                        if len(batch) >= 1 and isinstance(batch[0], torch.Tensor):
                            if len(batch[0].shape) == 3 and batch[0].shape[1] == 80:
                                audio_features = batch[0]
                                logging.debug(f"Found audio features with shape {audio_features.shape}")
                        
                        # Extract video features (second element in batch tuple)
                        if len(batch) >= 2 and isinstance(batch[1], torch.Tensor):
                            if len(batch[1].shape) == 5 and batch[1].shape[2] == 3:  # [batch, frames, channels=3, height, width]
                                video_features = batch[1]
                                logging.debug(f"Found video features with shape {video_features.shape}")
                    
                    # Extract utterance IDs for this batch
                    batch_utt_ids = []
                    if batch_idx * args.batch_size < len(test_dataset):
                        end_idx = min((batch_idx + 1) * args.batch_size, len(test_dataset))
                        for j in range(batch_idx * args.batch_size, end_idx):
                            video_path, audio_path, audio_id = test_dataset.names[j]
                            batch_utt_ids.append(audio_id)
                    logging.debug(f"Batch utterance IDs: {batch_utt_ids}")
                    
                    # Create a dictionary of inputs for the model
                    inputs_for_model = {}
                    
                    # Add audio features to model inputs
                    if audio_features is not None:
                        inputs_for_model["audio"] = audio_features.to(args.device)
                        logging.debug(f"Added audio features with shape: {audio_features.shape}")
                    
                    # Add video features to model inputs 
                    if video_features is not None:
                        inputs_for_model["video"] = video_features.to(args.device)
                        logging.debug(f"Added video features with shape: {video_features.shape}")
                    
                    # Verify we have the right inputs for the modality
                    if args.modality == "audio" and "audio" not in inputs_for_model:
                        logging.warning("Audio modality selected but no valid audio features found in inputs")
                        continue
                    elif args.modality == "video" and "video" not in inputs_for_model:
                        logging.warning("Video modality selected but no valid video features found in inputs")
                        continue
                    elif args.modality == "both" and ("audio" not in inputs_for_model or "video" not in inputs_for_model):
                        logging.warning("Both modality selected but both audio and video features are required")
                        continue
                    
                    # Force using both modalities if available
                    effective_modality = args.modality
                    if args.modality == "both" and "audio" in inputs_for_model and "video" in inputs_for_model:
                        logging.debug("Using both audio and video modalities")
                    elif "audio" in inputs_for_model and "video" not in inputs_for_model:
                        logging.debug("Only audio available, falling back to audio-only mode")
                        effective_modality = "audio"
                    elif "video" in inputs_for_model and "audio" not in inputs_for_model:
                        logging.debug("Only video available, falling back to video-only mode")
                        effective_modality = "video"
                    
                    logging.debug(f"Effective modality: {effective_modality}")
                    logging.debug(f"Model inputs: {list(inputs_for_model.keys())}")
                    
                    # Update progress bar
                    progress_bar.update(1)
                    
                    # Generate output using the model
                    outputs = model.generate(
                        audio=inputs_for_model.get("audio", None),
                        video=inputs_for_model.get("video", None),
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature
                    )

                    # Log number of outputs generated
                    num_outputs = len(outputs)
                    logging.debug(f"Generated {num_outputs} outputs")
                    
                    # Decode the output
                    decoded_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    logging.debug(f"Sample output: {decoded_text[0] if decoded_text else ''}")
                    
                    # Verify that we have the right number of batch_utt_ids
                    if len(batch_utt_ids) != num_outputs:
                        logging.warning(f"Mismatch between batch utterance IDs ({len(batch_utt_ids)}) and outputs ({num_outputs})")
                        # If needed, pad batch_utt_ids or truncate to match outputs
                        if len(batch_utt_ids) < num_outputs:
                            for idx in range(len(batch_utt_ids), num_outputs):
                                batch_utt_ids.append(f"unknown_{batch_idx}_{idx}")
                        elif len(batch_utt_ids) > num_outputs:
                            batch_utt_ids = batch_utt_ids[:num_outputs]
                    
                    # Record the results
                    for j, (output_tokens, text) in enumerate(zip(outputs, decoded_text)):
                        # Get the utterance ID for this sample
                        utt_id = batch_utt_ids[j] if j < len(batch_utt_ids) else f"unknown_{batch_idx}_{j}"
                        hypotheses[utt_id] = text.strip()
                        logging.debug(f"Added hypothesis for {utt_id}: {text.strip()}")
                        
                        # Store the reference if available
                        if utt_id in references:
                            ref_text = references[utt_id]
                            references_matched[utt_id] = ref_text
                            logging.debug(f"Found reference for {utt_id}: {ref_text}")
                            
                            # Print minimal output with just ID, HYP and REF
                            print(f"\nUTT: {utt_id}")
                            print(f"HYP: {text.strip()}")
                            print(f"REF: {ref_text}")
                            print("-" * 40)
                            
                            # Also collect for WER calculation
                            all_hypotheses.append(text.strip())
                            all_references.append(ref_text)
                            
                            # Record detailed result
                            results.append({
                                'utt_id': utt_id,
                                'hypothesis': text.strip(),
                                'reference': ref_text,
                                'wer': calculate_wer([ref_text], [text.strip()])
                            })
                        
                        # If we can't find a matching reference ID, try simplified ID
                        elif "/" in utt_id:
                            # Extract just the last part of the path as a possible ID
                            simple_id = utt_id.split("/")[-1]
                            if simple_id in references:
                                ref_text = references[simple_id]
                                references_matched[utt_id] = ref_text
                                logging.debug(f"Found reference using simplified ID: {simple_id} -> {ref_text}")
                                
                                # Print minimal output with just ID, HYP and REF
                                print(f"\nUTT: {utt_id}")
                                print(f"HYP: {text.strip()}")
                                print(f"REF: {ref_text}")
                                print("-" * 40)
                                
                                # Also collect for WER calculation
                                all_hypotheses.append(text.strip())
                                all_references.append(ref_text)
                                
                                # Record detailed result
                                results.append({
                                    'utt_id': utt_id,
                                    'hypothesis': text.strip(),
                                    'reference': ref_text,
                                    'wer': calculate_wer([ref_text], [text.strip()])
                                })
                            else:
                                logging.debug(f"No reference found for utterance ID: {utt_id} or simplified ID: {simple_id}")
                                print(f"\nUTT: {utt_id}")
                                print(f"HYP: {text.strip()}")
                                print(f"REF: [None]")
                                print("-" * 40)
                        else:
                            logging.debug(f"No reference found for utterance ID: {utt_id}")
                            print(f"\nUTT: {utt_id}")
                            print(f"HYP: {text.strip()}")
                            print(f"REF: [None]")
                            print("-" * 40)
                    
                    # Clean up to avoid CUDA OOM issues - do this after processing all samples in the batch
                    if 'outputs' in locals():
                        del outputs
                    if 'decoded_text' in locals():
                        del decoded_text
                    if 'inputs_for_model' in locals():
                        del inputs_for_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"Error processing batch {batch_idx}: {e}")
                    logging.error(traceback.format_exc())
                    continue
            
            # Close progress bar
            progress_bar.close()
                
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
                print(f"Modality: {args.modality}")
                print(f"Overall WER: {overall_wer:.4f}")
                print(f"Total samples: {len(all_references)}")
                print("="*80)
            else:
                logging.warning("No samples were successfully processed for WER calculation")
                print("\nNo samples were successfully processed for WER calculation")
        except Exception as e:
            logging.error(f"Error during decoding loop: {e}")
            logging.error(traceback.format_exc())
            print(f"\nError during decoding: {e}")
    
    except Exception as e:
        logging.error(f"Error during decoding: {e}")
        logging.error(traceback.format_exc())
        print(f"\nError during decoding: {e}")
        return 1

if __name__ == "__main__":
    main() 