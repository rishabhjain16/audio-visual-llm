import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
import os

def extract_audio_from_video(video_path, output_path=None, sample_rate=16000):
    """Extract audio from video file
    
    Args:
        video_path: Path to video file
        output_path: Path to save audio file (None to return without saving)
        sample_rate: Target sample rate
        
    Returns:
        audio (np.ndarray): Audio waveform
        sr (int): Sample rate
    """
    import subprocess
    import tempfile
    
    # Create temporary file if output path not specified
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

def extract_audio_features(audio_path, feature_type="mel", n_mels=80, hop_length=10, sample_rate=16000):
    """Extract audio features
    
    Args:
        audio_path: Path to audio file or numpy array of audio waveform
        feature_type: Type of features to extract ("mel", "mfcc", "raw")
        n_mels: Number of mel bands
        hop_length: Hop length in milliseconds
        sample_rate: Target sample rate
        
    Returns:
        Array of audio features
    """
    # Determine if input is a file path or audio array
    if isinstance(audio_path, str):
        # Load audio from file
        try:
            audio, sr = librosa.load(audio_path, sr=sample_rate)
        except:
            # Try extracting from video if audio loading fails
            try:
                audio, sr = extract_audio_from_video(audio_path, sample_rate=sample_rate)
            except Exception as e:
                print(f"Error extracting audio from {audio_path}: {e}")
                # Return empty features
                if feature_type == "raw":
                    return np.zeros(1000)
                else:
                    return np.zeros((100, n_mels))
    else:
        # Use provided audio array
        audio = audio_path
        sr = sample_rate
    
    # Convert hop length from ms to samples
    hop_length_samples = int(sr * hop_length / 1000)
    
    # Extract features based on type
    if feature_type == "mel":
        # Extract log mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=int(sr * 0.025),  # 25ms window
            hop_length=hop_length_samples
        )
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # Transpose to get (time, features) shape
        return log_mel_spec.T
    
    elif feature_type == "mfcc":
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=n_mels,  # Use n_mels as number of coefficients
            n_fft=int(sr * 0.025), 
            hop_length=hop_length_samples
        )
        # Add delta and delta-delta features
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        # Combine and transpose
        features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
        return features.T
    
    elif feature_type == "raw":
        # Just return raw audio
        return audio
    
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

def augment_audio(audio, sample_rate=16000, noise_factor=0.0, shift_factor=0.0, pitch_factor=0.0, speed_factor=0.0):
    """Apply augmentations to audio
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        noise_factor: Amount of noise to add
        shift_factor: Amount of time shift
        pitch_factor: Amount of pitch shift
        speed_factor: Amount of speed change
        
    Returns:
        Augmented audio waveform
    """
    augmented = audio.copy()
    
    # Add noise
    if noise_factor > 0:
        noise = np.random.randn(len(augmented))
        augmented += noise_factor * noise
    
    # Apply time shift
    if shift_factor > 0:
        shift = int(sample_rate * shift_factor)
        direction = np.random.choice([-1, 1])
        if direction == 1:
            # Shift right
            augmented = np.pad(augmented, (shift, 0), mode="constant")[:-shift]
        else:
            # Shift left
            augmented = np.pad(augmented, (0, shift), mode="constant")[shift:]
    
    # Apply pitch shift
    if pitch_factor > 0:
        pitch_change = np.random.uniform(-pitch_factor, pitch_factor)
        augmented = librosa.effects.pitch_shift(augmented, sr=sample_rate, n_steps=pitch_change)
    
    # Apply speed change
    if speed_factor > 0:
        speed_change = np.random.uniform(1.0 - speed_factor, 1.0 + speed_factor)
        augmented = librosa.effects.time_stretch(augmented, rate=speed_change)
        
        # Adjust length to match original
        if len(augmented) > len(audio):
            augmented = augmented[:len(audio)]
        else:
            augmented = np.pad(augmented, (0, max(0, len(audio) - len(augmented))), mode="constant")
    
    return augmented