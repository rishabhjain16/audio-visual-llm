# AVHuBERT-Whisper Model

This module provides an implementation of a multimodal speech recognition model that combines:

- **AVHuBERT**: For processing visual information (lip movements)
- **Whisper**: For processing audio information
- **Language Model (LLM)**: For generating text transcriptions

## Directory Structure

```
src/avhubert_whisper/
├── data/            # Dataset implementations
├── models/          # Model implementations
│   ├── av_hubert.py            # AVHuBERT encoder
│   └── avhubert_whisper_model.py  # Combined model
├── trainer/         # Training utilities
└── utils/           # Utility functions
```

## Key Components

### AVHuBERTEncoder

A simplified implementation of the AVHuBERT encoder that doesn't require fairseq dependency. It's designed to process video frames and extract features suitable for speech recognition.

### AVHuBERTWhisperModel

A complete multimodal model that:

1. Encodes video with AVHuBERT
2. Encodes audio with Whisper
3. Fuses these modalities and feeds them to a language model
4. Generates transcription output

## Usage

```python
from src.avhubert_whisper.models import AVHuBERTWhisperModel

# Initialize the model
model = AVHuBERTWhisperModel(
    llm_path="meta-llama/Llama-2-7b-chat-hf",
    whisper_model="openai/whisper-medium",
    avhubert_path="path/to/avhubert/checkpoint.pt",
    modality="both"  # Options: "audio", "video", "both"
)

# Forward pass
outputs = model(audio=audio_tensor, video=video_tensor)

# Generate text
results = model.generate(audio=audio_tensor, video=video_tensor)
```

## Training

Instructions for training the model will be added in a future update. 