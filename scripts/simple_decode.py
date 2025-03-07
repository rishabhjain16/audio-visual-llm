#!/usr/bin/env python3
"""
Decode script for the SimpleAVSRModel that supports WER calculation.
This script loads a trained model and runs inference on test data.
"""

import os
import sys
import argparse
import logging
import torch
import datetime
import yaml
import pandas as pd
from tqdm import tqdm
import jiwer
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.simple_avsr import SimpleAVSRModel
from src.data.processor import AVSRProcessor
from src.data.dataloader import AVSRDataset, collate_fn

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
    parser = argparse.ArgumentParser(description="Run inference with the SimpleAVSRModel and calculate WER")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data TSV file")
    parser.add_argument("--test_wrd", type=str, required=True, help="Path to test word reference file")
    parser.add_argument("--output_dir", type=str, default="outputs/decoding", help="Directory to save decoding results")
    parser.add_argument("--modality", type=str, default="both", choices=["audio", "video", "both"], 
                      help="Modality to use for inference")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="Device to run inference on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/simple.yaml", 
                      help="Configuration file for processor settings")
    parser.add_argument("--single_file", type=str, default=None, 
                      help="Path to single audio/video file for testing (overrides test_data)")
    
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
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Load model
    logging.info(f"Loading model from {args.model_path}")
    model = SimpleAVSRModel.from_pretrained(args.model_path)
    
    # Important: Set the modality explicitly from the command line args
    model.modality = args.modality
    logging.info(f"Using modality: {args.modality}")
    print(f"Using modality: {args.modality}")
    
    model.to(args.device)
    model.eval()
    
    # Load processor
    processor = AVSRProcessor.from_config(config)
    
    # Handle single file testing
    if args.single_file:
        logging.info(f"Testing single file: {args.single_file}")
        
        # Determine if it's audio or video based on extension
        file_ext = os.path.splitext(args.single_file)[1].lower()
        
        inputs = {}
        
        if file_ext in ['.wav', '.mp3', '.flac'] and args.modality in ["audio", "both"]:
            logging.info(f"Processing audio file: {args.single_file}")
            audio = processor.process_audio(args.single_file)
            inputs["audio"] = audio.unsqueeze(0).to(args.device)  # Add batch dimension
            
            if args.modality == "both":
                logging.warning("Using audio-only input with 'both' modality. No video input provided.")
        
        elif file_ext in ['.mp4', '.avi', '.mov'] and args.modality in ["video", "both"]:
            logging.info(f"Processing video file: {args.single_file}")
            video = processor.process_video(args.single_file)
            inputs["video"] = video.unsqueeze(0).to(args.device)  # Add batch dimension
            
            if args.modality == "both":
                logging.warning("Using video-only input with 'both' modality. No audio input provided.")
        
        else:
            logging.error(f"File type {file_ext} not supported or doesn't match modality {args.modality}")
            print(f"ERROR: File type {file_ext} not supported or doesn't match modality {args.modality}")
            return
        
        # Generate text
        logging.info("Generating text...")
        with torch.no_grad():
            try:
                # First get the embeddings from audio/video
                inputs_for_model = {}
                if "audio" in inputs and args.modality in ["audio", "both"]:
                    inputs_for_model["audio"] = inputs["audio"]
                if "video" in inputs and args.modality in ["video", "both"]:
                    inputs_for_model["video"] = inputs["video"]
                    
                # Ensure we have the right inputs for the selected modality
                if not inputs_for_model:
                    logging.error(f"No valid inputs for modality: {args.modality}")
                    return
                
                encoder_out = model(
                    **inputs_for_model,
                    return_loss=False,
                )
                
                # Generate text from embeddings
                generation_output = model.llm.generate(
                    inputs_embeds=encoder_out.hidden_states,
                    attention_mask=torch.ones(
                        (1, encoder_out.hidden_states.size(1)),
                        dtype=torch.long,
                        device=args.device
                    ),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                
                # Decode the output tokens
                generated_text = model.tokenizer.decode(generation_output[0], skip_special_tokens=True)
                
                logging.info(f"Generated text: {generated_text}")
                print("\nGenerated Text:")
                print(generated_text)
            except Exception as e:
                logging.error(f"Error during generation: {e}")
                logging.error(traceback.format_exc())
                print(f"Error during generation: {e}")
        
        return
    
    # Load test data
    logging.info(f"Loading test data from {args.test_data}")
    
    # Read test.wrd for reference texts
    references = {}
    try:
        with open(args.test_wrd, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        utt_id = parts[0]
                        ref_text = parts[1]
                        references[utt_id] = ref_text
        
        logging.info(f"Loaded {len(references)} reference transcriptions")
    except Exception as e:
        logging.error(f"Error loading references: {e}")
        return
    
    # Create test dataset
    try:
        test_dataset = AVSRDataset(
            data_path=args.test_data,
            processor=processor,
            split="test",
            modality=args.modality,
        )
        
        # Create dataloader
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
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
            utt_ids = batch["utt_id"]
            
            # Prepare inputs based on modality
            inputs_for_model = {}
            if "audio" in batch and args.modality in ["audio", "both"]:
                inputs_for_model["audio"] = batch["audio"].to(args.device)
            if "video" in batch and args.modality in ["video", "both"]:
                inputs_for_model["video"] = batch["video"].to(args.device)
            
            # Skip if we don't have the right inputs for the selected modality
            if not inputs_for_model:
                logging.warning(f"Batch {batch_idx} has no valid inputs for modality {args.modality}, skipping")
                continue
            
            with torch.no_grad():
                # Get encoder outputs
                encoder_out = model(
                    **inputs_for_model,
                    return_loss=False
                )
                
                # Generate text
                generation_output = model.llm.generate(
                    inputs_embeds=encoder_out.hidden_states,
                    attention_mask=torch.ones(
                        (encoder_out.hidden_states.size(0), encoder_out.hidden_states.size(1)),
                        dtype=torch.long, 
                        device=args.device
                    ),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,  # Deterministic for evaluation
                    num_beams=5,      # Beam search for better results
                )
                
                # Decode the output tokens for each sample
                batch_size = generation_output.size(0)
                for i in range(batch_size):
                    utt_id = utt_ids[i]
                    hypothesis = model.tokenizer.decode(generation_output[i], skip_special_tokens=True)
                    
                    # Get reference
                    reference = references.get(utt_id, "")
                    
                    # Calculate WER for this sample
                    if reference:
                        sample_wer = calculate_wer([reference], [hypothesis])
                        all_references.append(reference)
                        all_hypotheses.append(hypothesis)
                    else:
                        sample_wer = float('inf')
                        logging.warning(f"No reference found for utterance {utt_id}")
                    
                    # Save result
                    results.append({
                        "utt_id": utt_id,
                        "reference": reference,
                        "hypothesis": hypothesis,
                        "wer": sample_wer
                    })
        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {e}")
            logging.error(traceback.format_exc())
            continue
    
    # Calculate overall WER
    if all_references and all_hypotheses:
        overall_wer = calculate_wer(all_references, all_hypotheses)
        logging.info(f"Overall WER: {overall_wer:.4f}")
        
        # Save results
        with open(results_file, 'w') as f:
            f.write(f"Modality: {args.modality}\n")
            f.write(f"Overall WER: {overall_wer:.4f}\n\n")
            f.write(f"Detailed Results:\n")
            f.write(f"{'Utterance ID':<20} {'WER':<10} {'Reference':<40} {'Hypothesis':<40}\n")
            f.write("-" * 110 + "\n")
            
            for result in results:
                f.write(f"{result['utt_id']:<20} {result['wer']:<10.4f} {result['reference'][:40]:<40} {result['hypothesis'][:40]:<40}\n")
        
        logging.info(f"Results saved to {results_file}")
        print(f"\nModality: {args.modality}")
        print(f"Overall WER: {overall_wer:.4f}")
        print(f"Detailed results saved to {results_file}")
    else:
        logging.error("No valid results collected. Cannot calculate WER.")
        print("No valid results collected. Cannot calculate WER.")

if __name__ == "__main__":
    main() 