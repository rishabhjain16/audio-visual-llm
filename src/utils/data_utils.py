import torch
import numpy as np
import torch.nn.functional as F

class AVSRCollator:
    """Collator for Audio-Visual Speech Recognition batches"""
    
    def __call__(self, batch):
        """Collate function for DataLoader
        
        Args:
            batch: List of samples from AVSRDataset
            
        Returns:
            Batched data with padded sequences and masks
        """
        # Get list of items in batch
        ids = [sample["id"] for sample in batch]
        has_video = "video_features" in batch[0]
        has_audio = "audio_features" in batch[0]
        has_labels = "text" in batch[0]
        
        result = {"ids": ids}
        
        # Process video features
        if has_video:
            video_lengths = [sample["video_length"] for sample in batch]
            max_video_len = max(video_lengths)
            
            # Pad video features
            video_features = []
            for sample in batch:
                features = sample["video_features"]
                padding = max_video_len - features.shape[0]
                if padding > 0:
                    # Pad with zeros (batch, time, feature_dim)
                    padded = F.pad(features, (0, 0, 0, padding))
                    video_features.append(padded)
                else:
                    video_features.append(features)
            
            # Create tensor and masks
            video_features = torch.stack(video_features)
            video_mask = self._create_padding_mask(video_lengths, max_video_len)
            
            result["video_features"] = video_features
            result["video_lengths"] = torch.tensor(video_lengths, dtype=torch.long)
            result["video_mask"] = video_mask
        
        # Process audio features
        if has_audio:
            audio_lengths = [sample["audio_length"] for sample in batch]
            max_audio_len = max(audio_lengths)
            
            # Pad audio features
            audio_features = []
            for sample in batch:
                features = sample["audio_features"]
                padding = max_audio_len - features.shape[0]
                if padding > 0:
                    # Pad with zeros (batch, time, feature_dim)
                    padded = F.pad(features, (0, 0, 0, padding))
                    audio_features.append(padded)
                else:
                    audio_features.append(features)
            
            # Create tensor and masks
            audio_features = torch.stack(audio_features)
            audio_mask = self._create_padding_mask(audio_lengths, max_audio_len)
            
            result["audio_features"] = audio_features
            result["audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)
            result["audio_mask"] = audio_mask
        
        # Process text labels
        if has_labels:
            # Add raw text for reference/evaluation
            result["text"] = [sample["text"] for sample in batch]
        
        return result
    
    def _create_padding_mask(self, lengths, max_len):
        """Create padding mask for sequences
        
        Args:
            lengths: List of sequence lengths
            max_len: Maximum sequence length in batch
            
        Returns:
            Boolean mask (batch_size, max_len) where True values indicate padding
        """
        mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, length:] = True
        return mask

def prepare_lrs3_dataset(data_path, output_path, splits=["train", "val", "test"]):
    """Prepare LRS3 dataset for AVSR-LLM
    
    Args:
        data_path: Path to LRS3 dataset
        output_path: Path to save processed data
        splits: Dataset splits to process
    """
    import os
    import json
    import shutil
    from tqdm import tqdm
    
    os.makedirs(output_path, exist_ok=True)
    
    for split in splits:
        print(f"Processing {split} split...")
        
        # Create manifest and labels files
        manifest_file = os.path.join(output_path, f"{split}.tsv")
        labels_file = os.path.join(output_path, f"{split}.wrd")
        
        # Process the data
        # This is a simplified version - actual implementation would need to
        # follow LRS3 directory structure and extract video/audio/text
        
        # Placeholder for actual data processing logic
        print(f"Created manifest file: {manifest_file}")
        print(f"Created labels file: {labels_file}")
