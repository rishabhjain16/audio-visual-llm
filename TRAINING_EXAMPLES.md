# AVSR-LLM Training Examples

This document provides examples of how to train the AVSR-LLM model using different modalities and saving strategies.

## Training with Different Modalities

### Audio-Only Training

To train using only audio data:

```bash
./scripts/train_simple.sh \
  --data_path /path/to/your/data \
  --output_path outputs/audio_only_model \
  --modality audio \
  --batch_size 2 \
  --max_epochs 10
```

### Video-Only Training

To train using only video data:

```bash
./scripts/train_simple.sh \
  --data_path /path/to/your/data \
  --output_path outputs/video_only_model \
  --modality video \
  --batch_size 2 \
  --max_epochs 10
```

### Combined Training (Audio + Video)

To train using both audio and video data (default):

```bash
./scripts/train_simple.sh \
  --data_path /path/to/your/data \
  --output_path outputs/combined_model \
  --modality both \
  --batch_size 2 \
  --max_epochs 10
```

## Saving Strategies

### Epoch-Based Saving

Save a checkpoint every N epochs:

```bash
./scripts/train_simple.sh \
  --data_path /path/to/your/data \
  --save_every 2 \
  --max_epochs 10
```

### Step-Based Saving

Save a checkpoint every N training steps (parameter updates):

```bash
./scripts/train_simple.sh \
  --data_path /path/to/your/data \
  --save_steps 1000 \
  --max_epochs 10
```

## Monitoring Parameter Updates

To log parameter updates during training to verify learning is happening:

```bash
./scripts/train_simple.sh \
  --data_path /path/to/your/data \
  --log_param_updates \
  --max_epochs 10
```

## Complete Example

This example uses all the new features:

```bash
./scripts/train_simple.sh \
  --data_path /path/to/your/data \
  --output_path outputs/my_model \
  --llm_path checkpoints/Llama-3.2-1B \
  --modality audio \
  --batch_size 2 \
  --max_epochs 20 \
  --save_steps 500 \
  --log_param_updates
```

This will:
1. Train using audio data only
2. Save checkpoints every 500 training steps
3. Train for 20 epochs
4. Log parameter updates to verify learning
5. Use a batch size of 2 for stability

## Verifying Model Checkpoints

After training, you can find your model checkpoints in the specified output directory:

```
outputs/my_model/
  ├── checkpoint_step_500.pt    # Step-based checkpoint
  ├── checkpoint_step_1000.pt   # Step-based checkpoint
  ├── best_checkpoint.pt        # Best validation loss
  └── final_checkpoint.pt       # End of training
```

## Testing Your Trained Model

To test your trained model with different modalities, use the decoding script:

```bash
./scripts/simple_decode.sh \
  --model_path outputs/my_model \
  --test_data data/test.tsv \
  --test_wrd data/test.wrd \
  --modality audio
``` 