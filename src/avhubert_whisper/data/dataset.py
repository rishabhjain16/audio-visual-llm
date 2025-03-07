import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import librosa
from typing import Dict, List, Optional, Tuple, Any
import logging
import random

class AVSRDataset(Dataset):
    """Dataset for Audio-Visual Speech Recognition with pre-cropped lip region videos"""
    
    def __init__(
        self,
        manifest_path: str,
        label_path: Optional[str] = None,
        root_dir: Optional[str] = None,
        modalities: List[str] = ["audio", "video"],
        max_audio_length: int = 480000,  # 30s at 16kHz
        max_video_length: int = 600,     # ~24s at 25fps
        split: str = "train",
        audio_transform=None,
        video_transform=None,
        text_transform=None,
    ):
        """
        Args:
            manifest_path: Path to manifest TSV file with sample info
            label_path: Path to file with text labels
            root_dir: Root directory for paths in manifest (if relative)
            modalities: List of modalities to use ["audio", "video"]
            max_audio_length: Maximum audio length in samples
            max_video_length: Maximum video length in frames
            split: Data split (train, valid, test)
            audio_transform: Optional transform for audio
            video_transform: Optional transform for video
            text_transform: Optional transform for text labels
        """
        self.manifest_path = manifest_path
        self.label_path = label_path
        self.root_dir = root_dir or ""
        self.modalities = modalities
        self.max_audio_length = max_audio_length
        self.max_video_length = max_video_length
        self.split = split
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.text_transform = text_transform
        
        # Load manifest
        self.samples = self._load_manifest(manifest_path)
        
        # Load labels if provided
        self.labels = self._load_labels(label_path) if label_path else None
    
    def _load_manifest(self, manifest_path):
        """Load manifest file"""
        samples = []
        
        # First line of TSV is the root directory
        root_dir = None
        with open(manifest_path, "r") as f:
            first_line = f.readline().strip()
            if len(first_line) > 0 and not first_line.startswith("trainval") and not first_line.startswith("test"):
                root_dir = first_line
        
        # Read manifest file (TSV format)
        with open(manifest_path, "r") as f:
            # Skip the first line if it's the root directory
            if root_dir is not None:
                f.readline()
                
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:  # ID, video_path, audio_path, frames, samples
                    sample_id = parts[0]
                    video_path = parts[1]
                    audio_path = parts[2]
                    
                    # Check if paths are valid
                    if video_path and os.path.exists(video_path):
                        video_valid = True
                    else:
                        video_valid = False
                        
                    if audio_path and os.path.exists(audio_path):
                        audio_valid = True
                    else:
                        audio_valid = False
                    
                    # Only add sample if at least one modality is valid
                    if (video_valid and "video" in self.modalities) or (audio_valid and "audio" in self.modalities):
                        samples.append({
                            "id": sample_id,
                            "audio_path": audio_path if audio_valid else None,
                            "video_path": video_path if video_valid else None
                        })
        
        print(f"Loaded {len(samples)} samples from {manifest_path}")
        return samples
    
    def _load_labels(self, label_path):
        """Load text labels"""
        labels = {}
        
        # Read label file - one line per sample in the same order as manifest
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        # Map labels to sample IDs
        for i, sample in enumerate(self.samples):
            if i < len(lines):
                labels[sample["id"]] = lines[i].strip()
        
        print(f"Loaded {len(labels)} labels from {label_path}")
        return labels
    
    def _load_audio(self, audio_path):
        """Load audio file"""
        if audio_path is None or not os.path.exists(audio_path):
            return None, None
        
        try:
            # Load audio with librosa - these should be WAV files
            audio, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None, None
        
        # Truncate or pad
        if len(audio) > self.max_audio_length:
            audio = audio[:self.max_audio_length]
        elif len(audio) < self.max_audio_length:
            audio = np.pad(audio, (0, self.max_audio_length - len(audio)), "constant")
        
        return audio, sr
    
    def _load_video(self, video_path):
        """Load video frames - these should be pre-cropped lip region videos"""
        if video_path is None or not os.path.exists(video_path):
            return None
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            frames = []
            while len(frames) < self.max_video_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to grayscale to match VSR-LLM
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            
            cap.release()
            
            # Convert to numpy array
            if len(frames) > 0:
                frames = np.stack(frames)
                
                # Truncate or pad
                if len(frames) > self.max_video_length:
                    frames = frames[:self.max_video_length]
                elif len(frames) < self.max_video_length:
                    # Pad with zeros
                    padding = np.zeros((self.max_video_length - len(frames), *frames.shape[1:]), dtype=frames.dtype)
                    frames = np.concatenate([frames, padding], axis=0)
                
                return frames
            else:
                return None
        except Exception as e:
            print(f"Error loading video file {video_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample"""
        sample = self.samples[idx]
        sample_id = sample["id"]
        
        result = {"id": sample_id}
        
        # Load audio if needed
        if "audio" in self.modalities and sample["audio_path"]:
            audio, sr = self._load_audio(sample["audio_path"])
            if audio is not None:
                # Apply transform if available
                if self.audio_transform is not None:
                    audio = self.audio_transform(audio)
                
                result["audio"] = torch.FloatTensor(audio)
                result["sample_rate"] = sr
        
        # Load video if needed
        if "video" in self.modalities and sample["video_path"]:
            video = self._load_video(sample["video_path"])
            if video is not None:
                # Apply transform if available
                if self.video_transform is not None:
                    video = self.video_transform(video)
                
                result["video"] = torch.FloatTensor(video)
        
        # Add label if available
        if self.labels and sample_id in self.labels:
            text = self.labels[sample_id]
            
            # Apply transform if available
            if self.text_transform is not None:
                text = self.text_transform(text)
            
            result["text"] = text
        
        return result

class AVSRDataCollator:
    """Collator for batching AVSR samples"""
    
    def __call__(self, batch):
        """Collate batch"""
        batch_size = len(batch)
        
        # Initialize output dict
        output = {}
        
        # Get sample IDs
        output["id"] = [sample["id"] for sample in batch]
        
        # Process audio if available
        if "audio" in batch[0]:
            audio_lengths = [sample["audio"].size(0) for sample in batch]
            max_audio_len = max(audio_lengths)
            
            # Collate audio
            audio_tensor = torch.zeros(batch_size, max_audio_len)
            audio_padding_mask = torch.ones(batch_size, max_audio_len, dtype=torch.bool)
            
            for i, sample in enumerate(batch):
                audio = sample["audio"]
                audio_len = audio.size(0)
                
                audio_tensor[i, :audio_len] = audio
                audio_padding_mask[i, :audio_len] = False
            
            output["audio"] = audio_tensor
            output["audio_padding_mask"] = audio_padding_mask
            output["audio_lengths"] = torch.tensor(audio_lengths)
            
            if "sample_rate" in batch[0]:
                output["sample_rate"] = batch[0]["sample_rate"]
        
        # Process video if available
        if "video" in batch[0]:
            video_lengths = [sample["video"].size(0) for sample in batch]
            max_video_len = max(video_lengths)
            
            # Get video dimensions
            video_dims = batch[0]["video"].shape[1:]
            
            # Collate video
            video_tensor = torch.zeros(batch_size, max_video_len, *video_dims)
            video_padding_mask = torch.ones(batch_size, max_video_len, dtype=torch.bool)
            
            for i, sample in enumerate(batch):
                video = sample["video"]
                video_len = video.size(0)
                
                video_tensor[i, :video_len] = video
                video_padding_mask[i, :video_len] = False
            
            output["video"] = video_tensor
            output["video_padding_mask"] = video_padding_mask
            output["video_lengths"] = torch.tensor(video_lengths)
        
        # Process text if available
        if "text" in batch[0]:
            output["text"] = [sample["text"] for sample in batch]
        
        return output

def create_dataloader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=False,
):
    """Create a DataLoader for an AVSR dataset"""
    collator = AVSRDataCollator()
    
    # For debugging, use single worker
    if os.environ.get('AVSR_DEBUG', '0') == '1':
        num_workers = 0
        print("Debug mode: Using 0 workers for DataLoader")
    
    # Handle potential errors with num_workers
    try:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
            drop_last=False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
    except Exception as e:
        print(f"Error creating DataLoader with {num_workers} workers: {e}")
        print("Falling back to single-process loading")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collator,
            pin_memory=False,
            drop_last=False,
        )
class DummyDataset(Dataset):
    """
    A dummy dataset for testing AVSR-LLM model
    
    Generates random tensors for audio and video
    """
    
    def __init__(self, size=100, audio_dim=80, video_dim=512, seq_len=100, split="train"):
        """
        Initialize dummy dataset
        
        Args:
            size: Number of samples in the dataset
            audio_dim: Audio feature dimension
            video_dim: Video feature dimension
            seq_len: Sequence length
            split: Dataset split (train, val, test)
        """
        self.size = size
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.seq_len = seq_len
        self.split = split
        
        logging.info(f"Initialized {split} DummyDataset with {size} samples")
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random tensors
        audio = torch.randn(self.seq_len, self.audio_dim)
        video = torch.randn(self.seq_len, self.video_dim)
        
        # Generate random labels (10-30 tokens)
        label_len = random.randint(10, 30)
        labels = torch.randint(0, 1000, (label_len,))
        
        return {
            "audio": audio,
            "video": video,
            "labels": labels,
            "audio_len": self.seq_len,
            "video_len": self.seq_len,
            "label_len": label_len
        }
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader
        
        Args:
            batch: List of samples
            
        Returns:
            Batched tensors
        """
        # Stack audio and video
        audio = torch.stack([item["audio"] for item in batch])
        video = torch.stack([item["video"] for item in batch])
        
        # Pad labels to the same length
        max_label_len = max(item["label_len"] for item in batch)
        labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
        
        for i, item in enumerate(batch):
            labels[i, :item["label_len"]] = item["labels"]
        
        return {
            "audio": audio,
            "video": video,
            "labels": labels
        }

def create_datasets(config):
    """
    Create datasets for training and validation
    
    Args:
        config: Configuration object
        
    Returns:
        train_dataset, val_dataset
    """
    # Get dataset parameters from config
    audio_dim = getattr(config.data, "audio_dim", 80)
    video_dim = getattr(config.data, "video_dim", 512)
    seq_len = getattr(config.data, "max_seq_len", 100)
    
    # Create train dataset
    train_dataset = DummyDataset(
        size=100,
        audio_dim=audio_dim,
        video_dim=video_dim,
        seq_len=seq_len,
        split="train"
    )
    
    # Create validation dataset
    val_dataset = DummyDataset(
        size=20,
        audio_dim=audio_dim,
        video_dim=video_dim,
        seq_len=seq_len,
        split="val"
    )
    
    return train_dataset, val_dataset
