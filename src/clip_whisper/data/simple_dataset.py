import os
import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, CLIPProcessor, AutoTokenizer
from typing import List, Tuple, Dict, Any, Optional
import cv2
import soundfile as sf
import torch.nn.functional as F
import random
import traceback

class AVSRDataset(Dataset):
    """
    Audio-Visual Speech Recognition dataset based on VSR-LLM approach
    
    This dataset loads audio and video data from files in the LRS3 format
    and processes them for the model.
    """
    
    def __init__(
        self,
        manifest_path: str,
        label_path: str,
        root_dir: str,
        whisper_processor: WhisperProcessor,
        clip_processor: CLIPProcessor,
        tokenizer: Any,
        max_audio_length: int = 30,  # seconds
        max_video_length: int = 300,  # frames
        sampling_rate: int = 16000,
        split: str = "train",
        normalize: bool = True,
        image_mean: float = 0,
        image_std: float = 1,
        image_crop_size: int = 88,
        modality: str = "both",  # "audio", "video", or "both"
    ):
        super().__init__()
        
        self.root_dir = root_dir
        self.whisper_processor = whisper_processor
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.max_audio_length = max_audio_length
        self.max_video_length = max_video_length
        self.sampling_rate = sampling_rate
        self.split = split
        self.normalize = normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.image_crop_size = image_crop_size
        self.modality = modality  # Store which modality to use
        
        # Load manifest file
        logging.info(f"Loading manifest from {manifest_path}")
        self.names, self.sizes, total_entries = self._load_manifest(manifest_path)
        
        # Load labels
        logging.info(f"Loading labels from {label_path}")
        self.labels = self._load_labels(label_path)
        
        # If manifest and labels have different lengths, log the issue but don't trim anything
        if len(self.names) != len(self.labels):
            # Just log a warning about the mismatch but continue with all data
            logging.warning(f"Mismatch between manifest ({len(self.names)}) and labels ({len(self.labels)})")
            logging.warning("Using all available data without trimming.")
        
        logging.info(f"Dataset size: {len(self.names)} manifest entries, {len(self.labels)} label entries")
        logging.info(f"Dataset operating in '{self.modality}' modality mode")
    
    def _load_manifest(self, manifest_path):
        """Load manifest file in LRS3 format"""
        names = []
        sizes = []
        
        with open(manifest_path) as f:
            root = f.readline().strip()
            total_entries = 0
            
            for line in f:
                total_entries += 1
                items = line.strip().split("\t")
                if len(items) < 5:
                    logging.warning(f"Skipping invalid line: {line.strip()}")
                    continue
                
                # Format: id, video_path, audio_path, num_frames, num_samples
                audio_id = items[0]
                video_path = items[1]
                audio_path = items[2]
                
                # Try to parse frame count and sample count
                try:
                    num_frames = int(items[3])
                    num_samples = int(items[4])
                except (ValueError, IndexError):
                    logging.warning(f"Invalid frame/sample count in line: {line.strip()}")
                    continue
                
                # Keep ALL entries regardless of length
                names.append((video_path, audio_path, audio_id))
                sizes.append(num_samples)
        
        logging.info(f"Loaded {len(names)} entries from manifest file with {total_entries} total entries")
        return names, sizes, total_entries
    
    def _load_labels(self, label_path):
        """Load label file"""
        with open(label_path) as f:
            labels = [line.strip() for line in f]
        
        logging.info(f"Loaded {len(labels)} labels from label file")
        return labels
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        # Get paths
        video_path, audio_path, audio_id = self.names[idx]
        
        # Get full paths
        video_path = os.path.join(self.root_dir, video_path)
        audio_path = os.path.join(self.root_dir, audio_path)
        
        # Check if files exist
        audio_exists = os.path.exists(audio_path)
        video_exists = os.path.exists(video_path)
        
        if not audio_exists and not video_exists:
            logging.warning(f"Both audio and video files missing: {audio_path}, {video_path}")
            # Try another sample
            return self.__getitem__((idx + 1) % len(self))
        
        # Load audio if it exists and if we need audio features
        audio_features = None
        if audio_exists and (self.modality == "audio" or self.modality == "both"):
            try:
                # Load audio
                audio, sr = sf.read(audio_path)
                
                # Handle mono/stereo
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)  # Convert to mono
                
                # Handle normalization
                if self.normalize:
                    if np.abs(audio).max() > 1.0:
                        # Assume 16-bit audio
                        audio = audio.astype(np.float32) / 32768.0
                    else:
                        # Already normalized
                        audio = audio.astype(np.float32)
                
                # Process with whisper
                audio_features = self.whisper_processor(
                    audio, 
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt"
                ).input_features.squeeze(0)
                
                # Normalize if needed
                if self.normalize:
                    with torch.no_grad():
                        audio_features = F.layer_norm(audio_features, audio_features.shape)
                
            except Exception as e:
                logging.error(f"Error loading audio {audio_path}: {e}")
                audio_features = None
        
        # Load video if it exists and if we need video features
        video_features = None
        if video_exists and (self.modality == "video" or self.modality == "both"):
            try:
                # Open video file
                video = cv2.VideoCapture(video_path)
                frames = []
                
                # Read frames
                while len(frames) < self.max_video_length:
                    ret, frame = video.read()
                    if not ret:
                        break
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                
                video.release()
                
                if len(frames) == 0:
                    logging.warning(f"No frames found in video: {video_path}")
                    video_features = None
                else:
                    # Process with CLIP
                    processed_frames = []
                    for i in range(min(len(frames), self.max_video_length)):
                        # Use the CLIP processor to ensure proper formatting
                        try:
                            # Ensure the frame is in the correct format for CLIP
                            frame = frames[i]
                            
                            # Convert to RGB if not already
                            if frame.shape[-1] != 3:
                                logging.warning(f"Frame doesn't have 3 channels: {frame.shape}")
                                # Try to fix by assuming it's grayscale
                                if len(frame.shape) == 2:
                                    # Convert grayscale to RGB
                                    frame = np.stack([frame] * 3, axis=-1)
                            
                            # Ensure frame has proper dimensions
                            if len(frame.shape) != 3 or frame.shape[-1] != 3:
                                logging.warning(f"Skipping frame with invalid shape: {frame.shape}")
                                continue
                                
                            # Process with CLIP processor
                            with torch.no_grad():
                                processed = self.clip_processor(
                                    images=frame, 
                                    return_tensors="pt"
                                )
                                # CLIP expects pixel values in [B, C, H, W] format
                                pixel_values = processed["pixel_values"]
                                
                                # Verify shape is correct for CLIP
                                if pixel_values.dim() != 4 or pixel_values.size(1) != 3:
                                    logging.warning(f"Processed frame has unexpected shape: {pixel_values.shape}, expected [1, 3, H, W]")
                                
                                processed_frames.append(pixel_values)
                        except Exception as e:
                            logging.error(f"Error processing frame {i}: {e}")
                            continue
                    
                    if not processed_frames:
                        raise ValueError(f"Failed to process any video frames for {video_path}")
                        
                    # Stack frames properly (already in CLIP format)
                    # Each frame is [1, 3, 224, 224], we want [frames, 3, 224, 224]
                    video_features = torch.cat(processed_frames, dim=0)
                    
                    # Log the shape less frequently to reduce log spam
                    if random.random() < 0.05:  # Only log about 5% of the time
                        logging.info(f"Processed video shape: {video_features.shape}")
            
            except Exception as e:
                logging.error(f"Error loading video {video_path}: {e}")
                video_features = None
        
        # Handle required modalities not being available
        if self.modality == "audio" and audio_features is None:
            logging.warning(f"Audio required but not available for {audio_path}")
            return self.__getitem__((idx + 1) % len(self))
        elif self.modality == "video" and video_features is None:
            logging.warning(f"Video required but not available for {video_path}")
            return self.__getitem__((idx + 1) % len(self))
        elif self.modality == "both" and (audio_features is None or video_features is None):
            logging.warning(f"Both modalities required but not available for {audio_path}, {video_path}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Get text labels
        text = self.labels[idx]
        
        # Create tokens for model input
        # Ensure tokenizer has a pad token before tokenizing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Add the pad token to the vocabulary
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        
        # Get tokens for prompt and text
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": 256,  # Use a reasonable limit for labels
            "return_attention_mask": False
        }
        
        # Convert text to model tokens (for labels during training)
        tokens = self.tokenizer(text, **tokenizer_kwargs).input_ids[0].tolist()
        
        # Return data structured for the collate_fn
        return audio_features, video_features, text, tokens
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length sequences
        
        Args:
            batch: List of tuples (audio_features, video_features, text, label_tokens)
        
        Returns:
            Batched tensors with appropriate padding
        """
        try:
            # Extract components
            audio_features = []
            video_features = []
            texts = []
            label_tokens = []
            audio_shapes = []
            video_shapes = []
            
            # Track whether each modality exists in this batch
            has_audio = False
            has_video = False
            
            for item in batch:
                if len(item) != 4:
                    logging.warning(f"Invalid batch item: {item}")
                    continue
                    
                audio_feat, video_feat, text, labels = item
                
                # Check if audio features exist
                if audio_feat is not None:
                    audio_features.append(audio_feat)
                    audio_shapes.append((audio_feat.shape, audio_feat.dtype))
                    has_audio = True
                
                # Check if video features exist
                if video_feat is not None:
                    video_features.append(video_feat)
                    video_shapes.append((video_feat.shape, video_feat.dtype))
                    has_video = True
                
                texts.append(text)
                label_tokens.append(labels)
            
            # Log the shapes for debugging
            if audio_shapes:
                logging.debug(f"Audio shapes in batch: {audio_shapes}")
            if video_shapes:
                logging.debug(f"Video shapes in batch: {video_shapes}")
            
            # Collate audio features if they exist
            collated_audio = None
            if has_audio:
                # Check the dimension of the audio features
                if audio_features[0].dim() == 2:
                    # For 2D audio features [seq_len, feature_dim]
                    # Pad along the sequence dimension (dim=0)
                    max_audio_len = max(feat.size(0) for feat in audio_features)
                    feature_dim = audio_features[0].size(1)
                    
                    # Create padded batch
                    batch_size = len(audio_features)
                    collated_audio = torch.zeros(
                        batch_size, max_audio_len, feature_dim,
                        dtype=audio_features[0].dtype
                    )
                    
                    # Fill in the actual features
                    for i, feat in enumerate(audio_features):
                        seq_len = feat.size(0)
                        # Add batch dimension
                        collated_audio[i, :seq_len, :] = feat
                
                elif audio_features[0].dim() == 3:
                    # For 3D audio features [batch, seq_len, feature_dim]
                    # Pad along the sequence dimension (dim=1)
                    max_audio_len = max(feat.size(1) for feat in audio_features)
                    batch_size = len(audio_features)
                    feature_dim = audio_features[0].size(2)
                    
                    # Create tensor with appropriate shape filled with zeros
                    collated_audio = torch.zeros(
                        batch_size, max_audio_len, feature_dim,
                        dtype=audio_features[0].dtype
                    )
                    
                    # Fill in the actual features
                    for i, feat in enumerate(audio_features):
                        seq_len = feat.size(1)
                        collated_audio[i, :seq_len, :] = feat
                else:
                    logging.error(f"Unexpected audio feature dimension: {audio_features[0].dim()}")
                    raise ValueError(f"Unexpected audio feature dimension: {audio_features[0].dim()}")
            
            # Collate video features if they exist
            collated_video = None
            if has_video:
                # Get the maximum number of frames across the batch
                max_frames = max(feat.size(0) for feat in video_features)
                
                # Get the dimensions from the first video
                if video_features[0].dim() >= 4:  # [frames, channels, height, width]
                    channels, height, width = video_features[0].size()[1:]
                    batch_size = len(video_features)
                    
                    # Create padded tensor
                    collated_video = torch.zeros(
                        batch_size, max_frames, channels, height, width,
                        dtype=video_features[0].dtype
                    )
                    
                    # Fill in the features
                    for i, feat in enumerate(video_features):
                        frames = feat.size(0)
                        collated_video[i, :frames, :, :, :] = feat
                    
                    # Log the final shape
                    logging.debug(f"Final collated video shape: {collated_video.shape}")
                else:
                    logging.error(f"Unexpected video feature dimension: {video_features[0].dim()}")
                    raise ValueError(f"Unexpected video feature dimension: {video_features[0].dim()}")
            
            # Collate label tokens
            max_label_len = max(len(tokens) for tokens in label_tokens)
            
            # Create padded tensor for labels
            collated_labels = torch.full(
                (len(label_tokens), max_label_len),
                -100,  # Padding token ID (will be ignored in loss calculation)
                dtype=torch.long
            )
            
            # Fill in the actual tokens
            for i, tokens in enumerate(label_tokens):
                collated_labels[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            
            return collated_audio, collated_video, texts, collated_labels
        
        except Exception as e:
            logging.error(f"Error in collate_fn: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise e


def create_dataloaders(
    manifest_path: str = None,
    label_path: str = None,
    root_dir: str = None,
    data_path: str = None,
    whisper_processor: WhisperProcessor = None,
    clip_processor: CLIPProcessor = None,
    tokenizer: Any = None,
    batch_size: int = 8,
    num_workers: int = 4,
    max_audio_length: int = 30,
    max_video_length: int = 300,
    split: str = "train",
    config: dict = None,
):
    """Create dataloaders for training and validation
    
    Args:
        manifest_path: Path to manifest file
        label_path: Path to label file
        root_dir: Root directory for data
        data_path: Alternative way to specify data location (will derive manifest_path, label_path, and root_dir)
        whisper_processor: Processor for whisper model
        clip_processor: Processor for CLIP model
        tokenizer: Tokenizer for text
        batch_size: Batch size
        num_workers: Number of workers for dataloader
        max_audio_length: Maximum audio length in seconds
        max_video_length: Maximum video length in frames
        split: Dataset split (train, val, test)
        config: Configuration dictionary (overrides hardcoded file names)
    """
    
    # If data_path is provided, construct the other paths
    if data_path:
        root_dir = data_path
        
        # Get file names from config if available
        if config and "data" in config:
            data_config = config["data"]
            train_manifest_filename = data_config.get("train_manifest", "train.tsv")
            train_label_filename = data_config.get("train_labels", "train.wrd")
            val_manifest_filename = data_config.get("val_manifest", "valid.tsv")
            val_label_filename = data_config.get("val_labels", "valid.wrd")
        else:
            # Default file names
            train_manifest_filename = "train.tsv"
            train_label_filename = "train.wrd"
            val_manifest_filename = "valid.tsv"
            val_label_filename = "valid.wrd"
        
        # Try different directory structures
        # First, check if files are in the root directory
        train_manifest_path = os.path.join(data_path, train_manifest_filename)
        train_label_path = os.path.join(data_path, train_label_filename)
        val_manifest_path = os.path.join(data_path, val_manifest_filename)
        val_label_path = os.path.join(data_path, val_label_filename)
        
        # If files aren't in the root, check if they're in a 'train' subdirectory
        if not os.path.exists(train_manifest_path) or not os.path.exists(train_label_path):
            train_dir = os.path.join(data_path, "train")
            if os.path.exists(train_dir) and os.path.isdir(train_dir):
                logging.info(f"Files not found in root directory, checking train subdirectory: {train_dir}")
                train_manifest_path = os.path.join(train_dir, train_manifest_filename)
                train_label_path = os.path.join(train_dir, train_label_filename)
        
        # If files aren't in the train directory, check if they're in a 'data' subdirectory
        if not os.path.exists(train_manifest_path) or not os.path.exists(train_label_path):
            data_dir = os.path.join(data_path, "data")
            if os.path.exists(data_dir) and os.path.isdir(data_dir):
                logging.info(f"Files not found in train subdirectory, checking data subdirectory: {data_dir}")
                train_manifest_path = os.path.join(data_dir, train_manifest_filename)
                train_label_path = os.path.join(data_dir, train_label_filename)
                val_manifest_path = os.path.join(data_dir, val_manifest_filename)
                val_label_path = os.path.join(data_dir, val_label_filename)
    else:
        train_manifest_path = manifest_path
        train_label_path = label_path
        val_manifest_path = manifest_path.replace("train", "val")
        val_label_path = label_path.replace("train", "val")
    
    # Check if training paths exist
    if not os.path.exists(train_manifest_path):
        logging.error(f"Train manifest file not found: {train_manifest_path}")
        raise FileNotFoundError(f"Train manifest file not found: {train_manifest_path}")
    
    if not os.path.exists(train_label_path):
        logging.error(f"Train label file not found: {train_label_path}")
        raise FileNotFoundError(f"Train label file not found: {train_label_path}")
    
    if not os.path.exists(root_dir):
        logging.error(f"Root directory not found: {root_dir}")
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    # Import processors if not provided
    if whisper_processor is None or clip_processor is None or tokenizer is None:
        logging.info("Processors not provided, loading default ones...")
        from transformers import WhisperProcessor, CLIPProcessor, AutoTokenizer
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        # Ensure tokenizer has a pad token
        if tokenizer.pad_token is None:
            logging.info("Setting pad_token = eos_token for tokenizer in dataset")
            tokenizer.pad_token = tokenizer.eos_token
            # Add the pad token to the vocabulary
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    # Get the modality from the config
    modality = "both"
    if config and "modality" in config:
        modality = config["modality"]
        logging.info(f"Using modality from config: {modality}")
    
    # Create training dataset
    logging.info(f"Creating training dataset from {train_manifest_path}...")
    train_dataset = AVSRDataset(
        manifest_path=train_manifest_path,
        label_path=train_label_path,
        root_dir=root_dir,
        whisper_processor=whisper_processor,
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        max_audio_length=max_audio_length,
        max_video_length=max_video_length,
        split="train",
        modality=modality,  # Pass the modality parameter
    )
    
    # Check if dataset is empty
    if len(train_dataset) == 0:
        logging.error(f"Training dataset is empty after filtering")
        raise ValueError(f"Training dataset is empty after filtering")
    
    # Create training dataloader
    # This is safer, especially with float16 data
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=AVSRDataset.collate_fn,
        pin_memory=False,  # Disable pin_memory to prevent CUDA errors
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    
    logging.info(f"Created training dataloader with {len(train_dataset)} samples")
    
    # Create validation dataloader if validation files exist
    val_dataloader = None
    if os.path.exists(val_manifest_path) and os.path.exists(val_label_path):
        logging.info(f"Creating validation dataset from {val_manifest_path}...")
        val_dataset = AVSRDataset(
            manifest_path=val_manifest_path,
            label_path=val_label_path,
            root_dir=root_dir,
            whisper_processor=whisper_processor,
            clip_processor=clip_processor,
            tokenizer=tokenizer,
            max_audio_length=max_audio_length,
            max_video_length=max_video_length,
            split="val",
            modality=modality,  # Pass the modality parameter
        )
        
        if len(val_dataset) > 0:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=AVSRDataset.collate_fn,
                pin_memory=False,  # Disable pin_memory to prevent CUDA errors
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
            )
            logging.info(f"Created validation dataloader with {len(val_dataset)} samples")
        else:
            logging.warning("Validation dataset is empty, skipping validation")
    else:
        logging.warning(f"Validation files not found, skipping validation")
    
    return train_dataloader, val_dataloader 