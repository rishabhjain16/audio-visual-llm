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

Our model supports three modality modes:

1. **Audio-Only Mode**: Only processes audio inputs through the audio encoder and connector.
2. **Video-Only Mode**: Only processes video inputs through the video encoder and connector.
3. **Combined Mode (both)**: Processes both audio and video, fusing the features for improved performance.

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

## Controlling Training Duration

You can control the number of training epochs with:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --max_epochs 10 \
  --save_every 2 \
  --log_param_updates
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