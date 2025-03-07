#!/usr/bin/env python3
"""
Decode script for the SimpleAVSRModel.
This script loads a trained model and runs inference on audio/video files.
"""

import os
import sys
import argparse
import logging
import torch
import datetime
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.simple_avsr import SimpleAVSRModel
from src.data.processor import AVSRProcessor
from src.data.dataloader import collate_fn_inference
from transformers import set_seed

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run inference with the SimpleAVSRModel")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--audio_file", type=str, default=None, help="Path to the audio file for inference")
    parser.add_argument("--video_file", type=str, default=None, help="Path to the video file for inference")
    parser.add_argument("--modality", type=str, default="both", choices=["audio", "video", "both"], 
                      help="Modality to use for inference")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="Device to run inference on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/simple.yaml", 
                      help="Configuration file for processor settings")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    log_dir = "outputs/decoding"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"decode_{timestamp}.log")
    
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
    
    # Check if at least one modality is provided
    if args.modality in ["audio", "both"] and args.audio_file is None:
        logging.error("Audio file must be provided when audio modality is used")
        return
    
    if args.modality in ["video", "both"] and args.video_file is None:
        logging.error("Video file must be provided when video modality is used")
        return
    
    # Load model
    logging.info(f"Loading model from {args.model_path}")
    model = SimpleAVSRModel.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    
    # Load processor
    processor = AVSRProcessor.from_config(config)
    
    # Prepare inputs
    inputs = {}
    
    if args.audio_file and args.modality in ["audio", "both"]:
        logging.info(f"Processing audio file: {args.audio_file}")
        audio = processor.process_audio(args.audio_file)
        inputs["audio"] = audio.unsqueeze(0).to(args.device)  # Add batch dimension
    
    if args.video_file and args.modality in ["video", "both"]:
        logging.info(f"Processing video file: {args.video_file}")
        video = processor.process_video(args.video_file)
        inputs["video"] = video.unsqueeze(0).to(args.device)  # Add batch dimension
    
    # Generate text
    logging.info("Generating text...")
    with torch.no_grad():
        # First get the embeddings from audio/video
        encoder_out = model(
            audio=inputs.get("audio"),
            video=inputs.get("video"),
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

if __name__ == "__main__":
    main() 