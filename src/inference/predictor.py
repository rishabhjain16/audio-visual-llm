import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import cv2
import librosa
import time
import logging
from typing import Dict, List, Optional, Union, Any

from ..models.avsr_llm import AVSRLLModel
from ..data.dataset import AVSRDataset, create_dataloader
from ..preprocessing.data_prep import extract_frames, extract_audio_from_video
from ..utils.media import load_video, load_audio

class AVSRPredictor:
    """Predictor for AVSR-LLM model"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 256,
        num_beams: int = 5,
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run inference on
            max_length: Maximum length of generated sequence
            num_beams: Number of beams for beam search
        """
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load model from checkpoint"""
        logging.info(f"Loading model from {self.model_path}")
        
        # Load model
        model = AVSRLLModel.from_pretrained(self.model_path)
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Run inference on video/audio
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            prompt: Prompt for generation
            
        Returns:
            Generated text
        """
        # Load video/audio
        video_features = None
        audio_features = None
        
        if video_path is not None:
            video_features = load_video(video_path)
            video_features = video_features.to(self.device)
            
        if audio_path is not None:
            audio_features = load_audio(audio_path)
            audio_features = audio_features.to(self.device)
            
        # Generate text
        output = self.model.generate(
            video_features=video_features,
            audio_features=audio_features,
            prompt=prompt,
            max_length=self.max_length,
            num_beams=self.num_beams,
        )
        
        # Decode output
        text = self.model.decode_output(output)
        
        return text

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with AVSR-LLM model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Number of beams for beam search")
    
    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_path", type=str,
                      help="Path to a video file")
    group.add_argument("--manifest_path", type=str,
                      help="Path to a manifest file")
    
    # Output arguments
    parser.add_argument("--output_path", type=str,
                        help="Path to save output")
    parser.add_argument("--label_path", type=str,
                        help="Path to label file (for evaluation)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--data_root", type=str,
                        help="Root directory for data paths")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Create predictor
    predictor = AVSRPredictor(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    
    # Run prediction
    if args.video_path:
        # Process a single video
        output_path = args.output_path or f"{os.path.splitext(args.video_path)[0]}_transcript.txt"
        prediction = predictor.predict(video_path=args.video_path)
        print(f"
Transcription: {prediction}")
    
    elif args.manifest_path:
        # Process a dataset
        # Determine which modalities to use based on the model
        modalities = []
        if predictor.model.use_audio:
            modalities.append("audio")
        if predictor.model.use_video:
            modalities.append("video")
        
        # Create dataset
        dataset = AVSRDataset(
            manifest_path=args.manifest_path,
            label_path=args.label_path,
            root_dir=args.data_root,
            modalities=modalities,
            split="test"
        )
        
        # Run prediction
        output_file = args.output_path or f"{os.path.splitext(args.manifest_path)[0]}_predictions.txt"
        results = predictor.predict_dataset(dataset, output_file)

if __name__ == "__main__":
    main()