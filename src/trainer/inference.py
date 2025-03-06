import os
import torch
import numpy as np
from argparse import ArgumentParser
from ..models.avsr_llm import AVSRLLModel
from ..utils.video_utils import extract_frames, extract_face_landmarks
from ..utils.audio_utils import extract_audio_features, extract_audio_from_video
from ..utils.config import AVSRConfig
import logging
from typing import Dict, List, Optional, Any

def load_model(model_path, device="cuda"):
    """Load a trained AVSR-LLM model
    
    Args:
        model_path: Path to the model directory
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Load model configuration
    import yaml
    from argparse import Namespace
    
    config_path = os.path.join(model_path, "hparams.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        args = Namespace(**config)
    else:
        raise ValueError(f"Config file not found at {config_path}")
    
    # Initialize model with config
    model = AVSRLLModel(
        modalities=args.modalities,
        visual_input_dim=args.visual_input_dim,
        audio_input_dim=args.audio_input_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_layers=args.encoder_layers,
        fusion_type=args.fusion_type,
        llm_name=os.path.join(model_path, "llm"),
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_8bit=args.use_8bit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_seq_len=args.max_seq_len,
        chunk_size=args.chunk_size,
        prompt_template=args.prompt_template
    )
    
    # Load encoder weights
    encoder_weights_path = os.path.join(model_path, "encoder_weights.pt")
    if os.path.exists(encoder_weights_path):
        encoder_weights = torch.load(encoder_weights_path, map_location=device)
        model.load_state_dict(encoder_weights, strict=False)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model

def process_video(video_path, model, output_path=None, chunk_size=None, use_audio=True, use_video=True):
    """Process a video file with the AVSR-LLM model
    
    Args:
        video_path: Path to the video file
        model: Loaded AVSR-LLM model
        output_path: Path to save the output
        chunk_size: Size of chunks to process (None for whole video)
        use_audio: Whether to use audio modality
        use_video: Whether to use video modality
        
    Returns:
        Transcription text
    """
    # Extract frames and landmarks
    if use_video:
        frames = extract_frames(video_path, max_frames=model.max_seq_len)
        landmarks = extract_face_landmarks(frames, feature_type="lip")
        video_features = torch.FloatTensor(landmarks).unsqueeze(0)  # Add batch dimension
    else:
        video_features = None
    
    # Extract audio features
    if use_audio:
        audio, sr = extract_audio_from_video(video_path)
        audio_features = extract_audio_features(audio, feature_type="mel")
        audio_features = torch.FloatTensor(audio_features).unsqueeze(0)  # Add batch dimension
    else:
        audio_features = None
    
    # Create batch
    batch = {}
    if video_features is not None:
        batch["video_features"] = video_features
        batch["video_length"] = torch.tensor([video_features.shape[1]])
    
    if audio_features is not None:
        batch["audio_features"] = audio_features
        batch["audio_length"] = torch.tensor([audio_features.shape[1]])
    
    # Process in chunks if needed
    if chunk_size is not None and (video_features is not None and video_features.shape[1] > chunk_size):
        # Process in chunks and concatenate results
        all_preds = []
        
        for i in range(0, video_features.shape[1], chunk_size):
            chunk_batch = {}
            
            # Extract chunk of video features
            if video_features is not None:
                end_idx = min(i + chunk_size, video_features.shape[1])
                chunk_batch["video_features"] = video_features[:, i:end_idx]
                chunk_batch["video_length"] = torch.tensor([end_idx - i])
            
            # Extract corresponding chunk of audio features
            if audio_features is not None:
                # Calculate corresponding audio indices
                # This assumes audio and video are aligned and have the same rate
                audio_i = i * audio_features.shape[1] // video_features.shape[1]
                audio_end = min(audio_i + chunk_size * audio_features.shape[1] // video_features.shape[1], 
                                audio_features.shape[1])
                chunk_batch["audio_features"] = audio_features[:, audio_i:audio_end]
                chunk_batch["audio_length"] = torch.tensor([audio_end - audio_i])
            
            # Process chunk
            with torch.no_grad():
                chunk_preds = model.predict(chunk_batch)
            
            all_preds.extend(chunk_preds)
        
        # Combine predictions (simple concatenation for now)
        transcription = " ".join(all_preds)
    else:
        # Process whole video at once
        with torch.no_grad():
            preds = model.predict(batch)
        
        transcription = preds[0]  # Take first batch item
    
    # Save output if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(transcription)
    
    return transcription

def process_dataset(manifest_path, model, output_dir, use_audio=True, use_video=True):
    """Process a dataset with the AVSR-LLM model
    
    Args:
        manifest_path: Path to the manifest file
        model: Loaded AVSR-LLM model
        output_dir: Directory to save outputs
        use_audio: Whether to use audio modality
        use_video: Whether to use video modality
        
    Returns:
        Dictionary of sample_id -> transcription
    """
    # Load manifest
    samples = []
    with open(manifest_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                sample_id = parts[0]
                video_path = parts[1] if parts[1] != "NA" else None
                audio_path = parts[2] if len(parts) > 2 and parts[2] != "NA" else None
                
                samples.append({
                    "id": sample_id,
                    "video_path": video_path,
                    "audio_path": audio_path
                })
    
    # Process each sample
    results = {}
    for sample in samples:
        sample_id = sample["id"]
        video_path = sample["video_path"]
        
        # Skip if no video path and video is required
        if use_video and not video_path:
            print(f"Skipping {sample_id}: No video path")
            continue
        
        # Process video
        output_path = os.path.join(output_dir, f"{sample_id}.txt")
        transcription = process_video(
            video_path,
            model,
            output_path=output_path,
            use_audio=use_audio,
            use_video=use_video
        )
        
        results[sample_id] = transcription
    
    return results

def main():
    """Main function for inference"""
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--video_path", type=str, help="Path to a video file for inference")
    parser.add_argument("--manifest_path", type=str, help="Path to a manifest file for batch inference")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--use_audio", action="store_true", help="Use audio modality")
    parser.add_argument("--use_video", action="store_true", help="Use video modality")
    parser.add_argument("--chunk_size", type=int, help="Size of chunks to process")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Set default modalities if none specified
    if not args.use_audio and not args.use_video:
        args.use_audio = True
        args.use_video = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, device=args.device)
    
    # Process video or dataset
    if args.video_path:
        # Process single video
        output_path = os.path.join(args.output_dir, os.path.basename(args.video_path).split(".")[0] + ".txt")
        transcription = process_video(
            args.video_path,
            model,
            output_path=output_path,
            chunk_size=args.chunk_size,
            use_audio=args.use_audio,
            use_video=args.use_video
        )
        print(f"Transcription: {transcription}")
    
    elif args.manifest_path:
        # Process dataset
        results = process_dataset(
            args.manifest_path,
            model,
            args.output_dir,
            use_audio=args.use_audio,
            use_video=args.use_video
        )
        print(f"Processed {len(results)} samples. Results saved to {args.output_dir}")
    
    else:
        print("Error: Either --video_path or --manifest_path must be specified")

class InferenceEngine:
    """Inference engine for AVSR-LLM model"""
    
    def __init__(self, config: AVSRConfig, checkpoint_path: str):
        """
        Initialize inference engine
        
        Args:
            config: Configuration object
            checkpoint_path: Path to model checkpoint
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load model from checkpoint"""
        logging.info(f"Loading model from {self.checkpoint_path}")
        
        # Create model
        model = AVSRLLModel(self.config.model)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def infer(
        self,
        audio_features=None,
        video_features=None,
        prompt=None,
        max_length=None,
        num_beams=None,
    ):
        """
        Run inference
        
        Args:
            audio_features: Audio features
            video_features: Video features
            prompt: Prompt for generation
            max_length: Maximum length of generated sequence
            num_beams: Number of beams for beam search
            
        Returns:
            Generated text
        """
        # Set default values
        if max_length is None:
            max_length = self.config.model.max_length
        if num_beams is None:
            num_beams = self.config.model.num_beams
        
        # Generate text
        output = self.model.generate(
            audio_features=audio_features,
            video_features=video_features,
            prompt=prompt,
            max_length=max_length,
            num_beams=num_beams,
        )
        
        # Decode output
        text = self.model.decode_output(output)
        
        return text

if __name__ == "__main__":
    main()