#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import sys
import argparse
import logging
import torch
from pathlib import Path
from tqdm import tqdm

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.inference.inference_engine import AVSRInferenceEngine
from src.utils.config import load_config
from src.utils.setup import setup_logging, setup_environment, setup_seed
from src.utils.media import load_video, load_audio, save_results


def parse_args():
    parser = argparse.ArgumentParser(description="AVSR-LLM Inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input video/audio file or directory of files")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save inference results")
    parser.add_argument("--mode", type=str, choices=["audio", "video", "av"], default="av",
                        help="Input modality mode: audio, video, or audio-visual (av)")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device(s) to use, e.g., '0'")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for beam search decoding")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens per batch")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def run_inference(engine, input_path, output_path, mode, beam_size):
    """
    Run inference on a single file or directory of files
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)

    if input_path.is_file():
        files = [input_path]
    else:
        # Get all video/audio files in directory
        if mode in ["video", "av"]:
            video_exts = [".mp4", ".avi", ".mov", ".mkv"]
            files = [f for f in input_path.glob("**/*") if f.suffix.lower() in video_exts]
        else:
            audio_exts = [".wav", ".mp3", ".flac", ".ogg"]
            files = [f for f in input_path.glob("**/*") if f.suffix.lower() in audio_exts]

    # Ensure we have files to process
    if not files:
        logging.error(f"No {'video' if mode in ['video', 'av'] else 'audio'} files found in {input_path}")
        return

    results = {}
    for file_path in tqdm(files, desc="Processing files"):
        rel_path = file_path.relative_to(input_path) if input_path.is_dir() else file_path.name
        logging.info(f"Processing {rel_path}")

        try:
            # Load media based on mode
            if mode == "audio":
                audio = load_audio(file_path)
                video = None
            elif mode == "video":
                audio = None
                video = load_video(file_path)
            else:  # "av" mode
                audio = load_audio(file_path)
                video = load_video(file_path)

            # Run inference
            transcription = engine.transcribe(
                audio=audio,
                video=video,
                beam_size=beam_size
            )
            
            # Save results
            results[str(rel_path)] = transcription
            logging.info(f"Transcription: {transcription}")
            
        except Exception as e:
            logging.error(f"Error processing {rel_path}: {e}")
            results[str(rel_path)] = {"error": str(e)}

    # Save all results to output file
    save_results(results, output_path / "transcriptions.json")
    logging.info(f"Results saved to {output_path / 'transcriptions.json'}")


def main():
    args = parse_args()

    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    os.makedirs(args.output, exist_ok=True)
    setup_logging(Path(args.output) / "inference.log")
    
    # Setup environment and seed
    setup_environment()
    setup_seed(args.seed)
    
    # Initialize inference engine
    engine = AVSRInferenceEngine(
        config=config,
        checkpoint_path=args.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Run inference
    run_inference(
        engine=engine,
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        beam_size=args.beam_size
    )


if __name__ == "__main__":
    main() 