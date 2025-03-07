# ClipWhisper: Audio-Visual Speech Recognition with CLIP and Whisper

This module implements an Audio-Visual Speech Recognition (AVSR) model that combines CLIP (for video), Whisper (for audio), and an LLM (e.g., Llama) for generating transcriptions.

## Features

- **Multi-modal processing**: Support for audio-only, video-only, or combined modes
- **Flexible configuration**: Easy configuration through YAML files
- **Parameter logging**: Clear visibility into model dimensions and active components
- **Word Error Rate (WER) calculation**: Automatic evaluation during decoding
- **Step-based checkpoint saving**: Control model saving based on steps or epochs

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face)
- CLIP, Whisper, and Llama models
- jiwer (for WER calculation)

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare your data in the required format (TSV for metadata, WRD for references)

## Directory Structure

```
src/clip_whisper/
  ├── models/
  │   ├── clip_whisper_model.py   # Main model implementation
  │   └── modality_connector.py   # Connector module for encoders
  └── trainer/
      └── clip_whisper_trainer.py # Training implementation
      
scripts/clip_whisper/
  ├── train.py       # Python training script
  ├── train.sh       # Bash training script
  ├── decode.py      # Python decoding script
  └── decode.sh      # Bash decoding script
```

## Usage

### Training

Train the model using different modalities:

#### Audio-Only Training

```bash
./scripts/clip_whisper/train.sh \
  --data_path /path/to/your/data \
  --output_path outputs/audio_only_model \
  --modality audio \
  --batch_size 2 \
  --max_epochs 10
```

#### Video-Only Training

```bash
./scripts/clip_whisper/train.sh \
  --data_path /path/to/your/data \
  --output_path outputs/video_only_model \
  --modality video \
  --batch_size 2 \
  --max_epochs 10
```

#### Combined Training (Audio + Video)

```bash
./scripts/clip_whisper/train.sh \
  --data_path /path/to/your/data \
  --output_path outputs/combined_model \
  --modality both \
  --batch_size 2 \
  --max_epochs 10
```

### Decoding

Decode using different modalities:

#### Audio-Only Decoding

```bash
./scripts/clip_whisper/decode.sh \
  --model_path outputs/clip_whisper \
  --test_data data/test.tsv \
  --test_wrd data/test.wrd \
  --modality audio \
  --output_dir outputs/decoding_audio
```

#### Video-Only Decoding

```bash
./scripts/clip_whisper/decode.sh \
  --model_path outputs/clip_whisper \
  --test_data data/test.tsv \
  --test_wrd data/test.wrd \
  --modality video \
  --output_dir outputs/decoding_video
```

#### Combined Decoding

```bash
./scripts/clip_whisper/decode.sh \
  --model_path outputs/clip_whisper \
  --test_data data/test.tsv \
  --test_wrd data/test.wrd \
  --modality both \
  --output_dir outputs/decoding_both
```

### Single File Testing

You can also test on a single audio or video file:

```bash
./scripts/clip_whisper/decode.sh \
  --model_path outputs/clip_whisper \
  --single_file path/to/audio.wav \
  --modality audio
```

## Configuration

The model can be configured using a YAML file (see `configs/clip_whisper.yaml`):

```yaml
# Configure data paths, model parameters, training settings, etc.
data:
  path: "/path/to/your/data"
  # ...
model:
  modality: "both"  # "audio", "video", or "both"
  # ...
training:
  num_epochs: 10
  # ...
```

## Advanced Options

### Parameter Update Logging

To verify that parameters are being updated during training:

```bash
./scripts/clip_whisper/train.sh \
  --data_path /path/to/your/data \
  --log_param_updates
```

### Step-Based Saving

Save checkpoints based on training steps instead of epochs:

```bash
./scripts/clip_whisper/train.sh \
  --data_path /path/to/your/data \
  --save_steps 1000
```

### Verbose Decoding

Get detailed information during decoding:

```bash
./scripts/clip_whisper/decode.sh \
  --model_path outputs/clip_whisper \
  --test_data data/test.tsv \
  --test_wrd data/test.wrd \
  --verbose
```

## Model Architecture

```
┌─────────────┐     ┌──────────────┐
│ Audio Input │     │ Video Input  │
└──────┬──────┘     └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│   Whisper    │    │     CLIP     │
│   Encoder    │    │    Encoder   │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│    Audio     │    │    Video     │
│  Connector   │    │  Connector   │
│  (to LLM dim)│    │  (to LLM dim)│
└──────┬───────┘    └──────┬───────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
         ┌───────────────┐
         │  Fusion Layer │ (Only for "both" modality)
         │  (weighted avg)│
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │   LLM Model   │
         │ (Llama-based) │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │Generated Text │
         └───────────────┘
```

## Troubleshooting

If you encounter NaN or Inf values during training:
1. Try reducing the learning rate
2. Use a smaller batch size
3. Set `use_fp16=False` in the model config for better stability
4. Make sure the input data doesn't contain NaN or Inf values 