#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import sys
import logging
import argparse
from pathlib import Path
import json
import random
import shutil
from tqdm import tqdm
import multiprocessing as mp

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.audio_processor import AudioProcessor
from src.utils.config import load_config
from src.utils.setup import setup_logging, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for AVSR-LLM training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed dataset")
    parser.add_argument("--transcription_file", type=str, required=True,
                        help="JSON file containing video_id to transcription mappings")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Percentage of data to use for validation (0-1)")
    parser.add_argument("--test_split", type=float, default=0.05,
                        help="Percentage of data to use for testing (0-1)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes for parallel processing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing processed files")
    return parser.parse_args()


def process_video(args):
    """Process a single video file to extract audio and visual features"""
    video_path, config, output_dir, overwrite = args
    
    try:
        # Setup processors
        video_processor = VideoProcessor(config)
        audio_processor = AudioProcessor(config)
        
        # Create output path based on original path
        video_id = video_path.stem
        output_subdir = output_dir / video_id
        
        # Skip if already processed and not overwriting
        if output_subdir.exists() and not overwrite:
            return video_id, True, "Already processed"
        
        # Create output directory
        os.makedirs(output_subdir, exist_ok=True)
        
        # Process video to extract visual features
        visual_features = video_processor.extract_features(video_path)
        visual_features_path = output_subdir / "visual_features.pt"
        video_processor.save_features(visual_features, visual_features_path)
        
        # Process audio to extract audio features
        audio_features = audio_processor.extract_features(video_path)
        audio_features_path = output_subdir / "audio_features.pt"
        audio_processor.save_features(audio_features, audio_features_path)
        
        # Save metadata
        metadata = {
            "video_id": video_id,
            "original_path": str(video_path),
            "duration": video_processor.get_duration(video_path),
            "fps": video_processor.get_fps(video_path),
            "visual_features_shape": visual_features.shape,
            "audio_features_shape": audio_features.shape,
            "processing_config": config.preprocessing
        }
        
        with open(output_subdir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        return video_id, True, "Successfully processed"
        
    except Exception as e:
        logging.error(f"Error processing {video_path}: {e}")
        return video_path.stem, False, str(e)


def create_dataset_splits(processed_videos, transcriptions, output_dir, val_split, test_split, seed):
    """Create train/val/test splits and manifest files"""
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Filter videos that have transcriptions
    valid_videos = [vid for vid in processed_videos if vid in transcriptions]
    logging.info(f"Found transcriptions for {len(valid_videos)}/{len(processed_videos)} processed videos")
    
    # Shuffle videos
    random.shuffle(valid_videos)
    
    # Calculate split sizes
    val_size = int(len(valid_videos) * val_split)
    test_size = int(len(valid_videos) * test_split)
    train_size = len(valid_videos) - val_size - test_size
    
    # Create splits
    train_videos = valid_videos[:train_size]
    val_videos = valid_videos[train_size:train_size+val_size]
    test_videos = valid_videos[train_size+val_size:]
    
    logging.info(f"Created dataset splits: {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test")
    
    # Create manifest files
    splits = {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos
    }
    
    for split_name, videos in splits.items():
        manifest = []
        for video_id in videos:
            manifest.append({
                "id": video_id,
                "audio_path": str(output_dir / video_id / "audio_features.pt"),
                "visual_path": str(output_dir / video_id / "visual_features.pt"),
                "text": transcriptions[video_id]
            })
        
        # Save manifest
        manifest_path = output_dir / f"{split_name}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        logging.info(f"Created {split_name} manifest with {len(manifest)} examples at {manifest_path}")


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_dir = output_dir / "logs"
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir / "prepare_dataset.log")
    
    # Set random seed
    setup_seed(args.seed)
    
    # Load transcriptions
    with open(args.transcription_file, "r") as f:
        transcriptions = json.load(f)
    
    logging.info(f"Loaded {len(transcriptions)} transcriptions from {args.transcription_file}")
    
    # Find all video files
    input_dir = Path(args.input_dir)
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(input_dir.glob(f"**/*{ext}")))
    
    logging.info(f"Found {len(video_files)} video files in {input_dir}")
    
    # Process videos in parallel
    process_args = [(video_path, config, output_dir, args.overwrite) for video_path in video_files]
    
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(process_video, process_args), total=len(process_args), desc="Processing videos"))
    
    # Collect results
    successful = [video_id for video_id, success, _ in results if success]
    failed = [(video_id, reason) for video_id, success, reason in results if not success]
    
    logging.info(f"Processed {len(successful)} videos successfully, {len(failed)} failed")
    
    # Log failed videos
    if failed:
        with open(log_dir / "failed_videos.json", "w") as f:
            json.dump(failed, f, indent=2)
        logging.warning(f"Failed videos written to {log_dir / 'failed_videos.json'}")
    
    # Create dataset splits
    create_dataset_splits(
        processed_videos=successful,
        transcriptions=transcriptions,
        output_dir=output_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )


if __name__ == "__main__":
    main() 