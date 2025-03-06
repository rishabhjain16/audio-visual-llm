import os
import sys
import glob
import subprocess
import numpy as np
import cv2
import torch
import librosa
import argparse
from tqdm import tqdm
from pathlib import Path
import json

def extract_audio_from_video(video_path, output_path=None, sample_rate=16000):
    """Extract audio from video file
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the audio file (if None, returns without saving)
        sample_rate: Target sample rate
        
    Returns:
        audio: Audio waveform
        sr: Sample rate
    """
    import tempfile
    
    # Create temporary file if output path not provided
    temp_file = None
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = temp_file.name
    
    # Use ffmpeg to extract audio
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        output_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio, sr = librosa.load(output_path, sr=sample_rate)
    finally:
        # Clean up temp file if created
        if temp_file is not None:
            os.unlink(temp_file.name)
    
    return audio, sr

def extract_frames(video_path, output_dir=None, fps=25, face_detection=True):
    """Extract frames from video
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames (if None, returns without saving)
        fps: Target frames per second
        face_detection: Whether to detect and crop faces
        
    Returns:
        frames: List of extracted frames
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval
    if fps and fps < original_fps:
        interval = max(1, round(original_fps / fps))
    else:
        interval = 1
    
    # Prepare for face detection if needed
    face_detector = None
    if face_detection:
        try:
            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except Exception as e:
            print(f"Warning: Failed to initialize face detector: {e}")
            face_detection = False
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    frames = []
    frame_idx = 0
    
    with tqdm(total=total_frames//interval, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frames at specified interval
            if frame_idx % interval == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect and crop face if requested
                if face_detection and face_detector is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    faces = face_detector.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        x, y, w, h = faces[0]  # Use the first face detected
                        
                        # Add some margin
                        margin = int(min(w, h) * 0.2)
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(frame.shape[1] - x, w + 2 * margin)
                        h = min(frame.shape[0] - y, h + 2 * margin)
                        
                        # Crop face region
                        face = frame[y:y+h, x:x+w]
                        
                        # Resize to standard size (112x112)
                        face = cv2.resize(face, (112, 112))
                        frame = face
                
                # Save frame if output directory provided
                if output_dir is not None:
                    frame_path = os.path.join(output_dir, f"frame_{len(frames):06d}.jpg")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                frames.append(frame)
                pbar.update(1)
            
            frame_idx += 1
    
    cap.release()
    return frames

def prepare_lrs3_dataset(data_dir, output_dir, splits=["train", "val", "test"]):
    """Prepare LRS3 dataset for AVSR-LLM
    
    Args:
        data_dir: Path to the LRS3 dataset directory
        output_dir: Directory to save processed data
        splits: Dataset splits to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths based on LRS3 structure
    video_pattern = os.path.join(data_dir, "*.mp4")
    text_pattern = os.path.join(data_dir, "*.txt")
    
    # Get all video files
    video_files = sorted(glob.glob(video_pattern))
    text_files = sorted(glob.glob(text_pattern))
    
    # Print info
    print(f"Found {len(video_files)} video files and {len(text_files)} text files")
    
    # Load and create mapping from text file
    text_mapping = {}
    
    for text_file in text_files:
        try:
            with open(text_file, "r") as f:
                sample_id = os.path.splitext(os.path.basename(text_file))[0]
                text = f.read().strip()
                text_mapping[sample_id] = text
        except Exception as e:
            print(f"Error reading {text_file}: {e}")
    
    # Split data if needed
    if len(splits) > 1:
        # Random split for demonstration
        np.random.seed(42)
        indices = np.random.permutation(len(video_files))
        
        # Define split boundaries (80/10/10 by default)
        train_end = int(0.8 * len(video_files))
        val_end = int(0.9 * len(video_files))
        
        split_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:]
        }
    else:
        # Use all data for a single split
        split_indices = {splits[0]: np.arange(len(video_files))}
    
    # Process each split
    for split in splits:
        if split not in split_indices:
            continue
        
        # Create manifest and labels files
        manifest_file = os.path.join(output_dir, f"{split}.tsv")
        label_file = os.path.join(output_dir, f"{split}.wrd")
        
        # Open files for writing
        with open(manifest_file, "w") as manifest_f, open(label_file, "w") as label_f:
            # Write header if needed
            # manifest_f.write("ID	AUDIO_PATH	VIDEO_PATH
")
            
            # Process each video
            for idx in tqdm(split_indices[split], desc=f"Processing {split} split"):
                video_file = video_files[idx]
                sample_id = os.path.splitext(os.path.basename(video_file))[0]
                
                # Skip if no text available
                if sample_id not in text_mapping:
                    print(f"Warning: No text found for {sample_id}, skipping")
                    continue
                
                # Get text
                text = text_mapping[sample_id]
                
                # Add to manifest
                manifest_f.write(f"{sample_id}	{video_file}
")
                
                # Add to labels
                label_f.write(f"{sample_id} {text}
")
        
        print(f"Created {split} manifest: {manifest_file}")
        print(f"Created {split} labels: {label_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Prepare data for AVSR-LLM")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to source data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--dataset", type=str, default="lrs3", choices=["lrs3", "custom"],
                        help="Dataset type")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                        help="Dataset splits to process")
    parser.add_argument("--extract_frames", action="store_true", help="Extract video frames")
    parser.add_argument("--extract_audio", action="store_true", help="Extract audio")
    parser.add_argument("--fps", type=int, default=25, help="Target frames per second")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target audio sample rate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process dataset based on type
    if args.dataset == "lrs3":
        prepare_lrs3_dataset(args.data_dir, args.output_dir, args.splits)
    else:
        # Custom dataset processing would go here
        print("Custom dataset processing not implemented yet")
    
    # Extract frames if requested
    if args.extract_frames:
        # Process each split
        for split in args.splits:
            manifest_file = os.path.join(args.output_dir, f"{split}.tsv")
            if not os.path.exists(manifest_file):
                print(f"Warning: {manifest_file} not found, skipping frame extraction")
                continue
            
            # Load manifest
            with open(manifest_file, "r") as f:
                lines = f.readlines()
            
            # Process each video
            for line in tqdm(lines, desc=f"Extracting frames for {split}"):
                parts = line.strip().split("	")
                if len(parts) >= 2:
                    sample_id = parts[0]
                    video_path = parts[1]
                    
                    # Extract frames
                    output_dir = os.path.join(args.output_dir, "frames", split, sample_id)
                    extract_frames(video_path, output_dir, args.fps)
    
    # Extract audio if requested
    if args.extract_audio:
        # Process each split
        for split in args.splits:
            manifest_file = os.path.join(args.output_dir, f"{split}.tsv")
            if not os.path.exists(manifest_file):
                print(f"Warning: {manifest_file} not found, skipping audio extraction")
                continue
            
            # Load manifest
            with open(manifest_file, "r") as f:
                lines = f.readlines()
            
            # Process each video
            for line in tqdm(lines, desc=f"Extracting audio for {split}"):
                parts = line.strip().split("	")
                if len(parts) >= 2:
                    sample_id = parts[0]
                    video_path = parts[1]
                    
                    # Extract audio
                    output_path = os.path.join(args.output_dir, "audio", split, f"{sample_id}.wav")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    extract_audio_from_video(video_path, output_path, args.sample_rate)

if __name__ == "__main__":
    main()