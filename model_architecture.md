# AVSR-LLM: Model Architecture & Process Flow

## Model Architecture

Our Audio-Visual Speech Recognition (AVSR) model integrates audio and video inputs to generate text transcriptions. Here's a simplified architecture diagram:

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
         │ (Llama-2-7b)  │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │Generated Text │
         └───────────────┘
```

## Key Components

1. **Audio Encoder (Whisper)**: Processes audio inputs and extracts features.
2. **Video Encoder (CLIP)**: Processes video inputs to extract visual features.
3. **Modality Connectors**: Projects each modality to the LLM dimension with consistent dtype (float32).
4. **Fusion Layer**: Only used in "both" modality, combines audio and video features with weighted averaging.
5. **LLM (Llama-2)**: Large Language Model that generates text from the encoded features.

## Data Flow During Training

```
┌─────────────┐        ┌─────────────┐
│   Training  │ ─────> │ Dataset     │
│     Data    │        │(audio/video)│
└─────────────┘        └──────┬──────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  Dataloader  │
                       └──────┬───────┘
                              │
                              ▼
┌──────────────────────────────────────────────┐
│                  Train Loop                  │
│  ┌──────────┐    ┌────────┐    ┌──────────┐  │
│  │  Forward │ -> │ Loss   │ -> │ Backward │  │
│  │  Pass    │    │ Calc   │    │ Pass     │  │
│  └──────────┘    └────────┘    └──────────┘  │
│          │                          │        │
│          └──────────────────────────┘        │
│                      │                        │
│                      ▼                        │
│              ┌───────────────┐                │
│              │  Parameter    │                │
│              │   Updates     │                │
│              └───────────────┘                │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
                ┌──────────────┐
                │  Validation  │
                │     Loop     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ Save Model   │
                │ Checkpoint   │
                └──────────────┘
```

## Modality Handling

Our model supports three modality modes, which can be selected for both training and inference:

1. **Audio-Only Mode**: Only processes audio inputs through the audio encoder and connector. Fusion is skipped as there's no video input.
2. **Video-Only Mode**: Only processes video inputs through the video encoder and connector. Fusion is skipped as there's no audio input.
3. **Combined Mode (both)**: Processes both audio and video, fusing the features for improved performance.

The model automatically detects whether to apply fusion based on:
- The selected modality
- The availability of inputs (audio and/or video)

If the modality is set to "both" but only one input type is available, the model will process just that input without fusion.

## Decoding Process

```
┌─────────────┐     ┌──────────────┐
│ Audio/Video │ --> │  Load Model  │
│    Input    │     │ Checkpoint   │
└─────────────┘     └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │Input Features│
                    │ Extraction   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Connector   │
                    │ Projection   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    LLM       │
                    │  Generation  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Transcription│
                    │   Output     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ WER          │
                    │ Calculation  │
                    └──────────────┘
```

## Parameter Updates During Training

When you enable parameter update logging with `--log_param_updates`, the system will track and log:

1. Gradient norms during backward passes
2. Parameter norms after updates
3. Current learning rate

This allows you to verify that training is actually updating the parameters properly.

## Saving Checkpoints

The model saves checkpoints based on:

1. **Regular intervals**: Every `save_every` epochs (default: 1) - configurable with `--save_every`
2. **Best performance**: When validation loss improves
3. **Final checkpoint**: At the end of training

## Controlling Training

### Modality Selection for Training

You can control which modality to use during training with the `--modality` parameter:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --modality audio  # Use only audio for training
```

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --modality video  # Use only video for training
```

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --modality both   # Use both audio and video (default)
```

### Checkpoint Saving Control

You can control checkpoint saving in two ways:

1. **Epoch-based saving** - Save checkpoints every N epochs:
```bash
python scripts/train.py \
  --config configs/default.yaml \
  --save_every 2  # Save every 2 epochs
```

2. **Step-based saving** - Save checkpoints every N steps (parameter updates):
```bash
python scripts/train.py \
  --config configs/default.yaml \
  --save_steps 1000  # Save every 1000 training steps
```

When using step-based saving, it overrides the epoch-based saving behavior.

### Training Duration Control

You can control how long to train with:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --max_epochs 10  # Train for 10 epochs
```

### Parameter Update Monitoring

To verify parameter updates are happening properly:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --log_param_updates  # Track and log parameter changes
```

## Memory Management

The model implements several memory optimization strategies:
- Gradient accumulation to handle larger batch sizes
- Automatic memory management to detect and prevent OOM errors
- Proper tensor cleanup to reduce memory fragmentation

## Modality Testing Command Examples

### Audio-Only Mode:
```bash
./scripts/simple_decode.sh \
  --model_path outputs/simple_avsr \
  --test_data data/test.tsv \
  --test_wrd data/test.wrd \
  --modality audio
```

### Video-Only Mode:
```bash
./scripts/simple_decode.sh \
  --model_path outputs/simple_avsr \
  --test_data data/test.tsv \
  --test_wrd data/test.wrd \
  --modality video
```

### Combined Mode:
```bash
./scripts/simple_decode.sh \
  --model_path outputs/simple_avsr \
  --test_data data/test.tsv \
  --test_wrd data/test.wrd \
  --modality both
``` 