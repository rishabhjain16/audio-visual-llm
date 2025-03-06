#!/usr/bin/env python3
# Copyright (c) 2023-2024
# All rights reserved.

import os
import json
import logging
import numpy as np
import torch
import torchaudio
import torchvision
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image


def load_audio(file_path: Union[str, Path]) -> torch.Tensor:
    """
    Load audio file and convert to tensor
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Audio tensor of shape (channels, time)
    """
    file_path = str(file_path)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    except Exception as e:
        # If torchaudio fails, try using ffmpeg
        logging.warning(f"Error loading audio with torchaudio: {e}")
        logging.warning(f"Trying to load with ffmpeg...")
        
        try:
            import subprocess
            import tempfile
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Convert to WAV using ffmpeg
            subprocess.run([
                "ffmpeg", "-i", file_path, "-ar", "16000", "-ac", "1", 
                "-y", temp_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Load WAV file
            waveform, sample_rate = torchaudio.load(temp_path)
            
            # Remove temporary file
            os.unlink(temp_path)
            
            return waveform
            
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio file: {e2}")


def load_video(file_path: Union[str, Path]) -> torch.Tensor:
    """
    Load video file and convert to tensor
    
    Args:
        file_path: Path to video file
        
    Returns:
        Video tensor of shape (time, channels, height, width)
    """
    file_path = str(file_path)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    try:
        # Open video file
        video_reader = cv2.VideoCapture(file_path)
        
        # Check if video opened successfully
        if not video_reader.isOpened():
            raise RuntimeError(f"Failed to open video file: {file_path}")
        
        # Get video properties
        frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize frames list
        frames = []
        
        # Read frames
        for _ in range(frame_count):
            ret, frame = video_reader.read()
            
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to torch tensor
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            frames.append(frame_tensor)
        
        # Release video reader
        video_reader.release()
        
        # Stack frames
        if len(frames) > 0:
            return torch.stack(frames)
        else:
            raise RuntimeError(f"No frames read from video file: {file_path}")
    
    except Exception as e:
        # If OpenCV fails, try using torchvision
        logging.warning(f"Error loading video with OpenCV: {e}")
        logging.warning(f"Trying to load with torchvision...")
        
        try:
            import av
            from torchvision.io import read_video
            
            # Read video using torchvision
            frames, _, _ = read_video(file_path)
            
            # Convert to float and normalize
            frames = frames.float() / 255.0
            
            # Permute dimensions from (T, H, W, C) to (T, C, H, W)
            frames = frames.permute(0, 3, 1, 2)
            
            return frames
            
        except Exception as e2:
            raise RuntimeError(f"Failed to load video file: {e2}")


def save_audio(waveform: torch.Tensor, file_path: Union[str, Path], sample_rate: int = 16000) -> None:
    """
    Save audio tensor to file
    
    Args:
        waveform: Audio tensor of shape (channels, time)
        file_path: Path to save audio file
        sample_rate: Sample rate of audio
    """
    file_path = Path(file_path)
    
    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save audio
    torchaudio.save(str(file_path), waveform, sample_rate)


def save_video(frames: torch.Tensor, file_path: Union[str, Path], fps: int = 25) -> None:
    """
    Save video tensor to file
    
    Args:
        frames: Video tensor of shape (time, channels, height, width)
        file_path: Path to save video file
        fps: Frames per second
    """
    file_path = Path(file_path)
    
    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy and scale to 0-255
    frames_np = (frames * 255.0).byte().permute(0, 2, 3, 1).numpy()
    
    # Get video properties
    frame_count, height, width, channels = frames_np.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames_np:
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    # Release video writer
    video_writer.release()


def extract_audio_from_video(video_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Union[torch.Tensor, str]:
    """
    Extract audio from video file
    
    Args:
        video_path: Path to video file
        output_path: Path to save extracted audio file (optional)
        
    Returns:
        Audio tensor if output_path is None, otherwise path to saved audio file
    """
    video_path = Path(video_path)
    
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)
    
    try:
        # Extract audio using ffmpeg
        import subprocess
        
        subprocess.run([
            "ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", "-y", str(output_path)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if output_path is None:
            # Load extracted audio
            return load_audio(output_path)
        else:
            return str(output_path)
    
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio from video: {e}")


def save_results(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save inference results to JSON file
    
    Args:
        results: Dictionary of inference results
        output_path: Path to save results
    """
    output_path = Path(output_path)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results to JSON file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logging.info(f"Results saved to {output_path}") 