# AVSR-LLM: Audio-Visual Speech Recognition with Large Language Models

This repository contains the implementation of AVSR-LLM, a framework for Audio-Visual Speech Recognition using Large Language Models. The codebase is designed to be modular, easy to understand, and extensible for research purposes.

## Features

- **Multimodal Speech Recognition**: Combines audio and visual modalities for improved speech recognition
- **LLM Integration**: Leverages the power of Large Language Models for speech understanding
- **Flexible Architecture**: Supports different audio encoders, visual encoders, and fusion methods
- **Efficient Training**: Parameter-efficient fine-tuning with LoRA
- **Easy Experimentation**: Modular design with configuration-based setup

## Installation

```bash
# Create conda environment
conda create -n avsr-llm python=3.9 -y
conda activate avsr-llm

# Clone repository
git clone https://github.com/yourusername/AVSR-LLM.git
cd AVSR-LLM

# Install dependencies
pip install -r requirements.txt
```

## Important Notes on Dependencies

### About fairseq dependency

The original AV-HuBERT model requires the fairseq library, but we've implemented a simplified version that doesn't require fairseq. Our implementation creates a compatible model structure that mimics the original AV-HuBERT architecture.

**Note**: If you want to use the original AV-HuBERT model with exact weights, you'll need to install fairseq. However, this may be challenging due to dependency conflicts. 

```bash
# Optional: Install fairseq (note that this might cause dependency conflicts)
# pip install fairseq
```

## Model Checkpoints

You need to download the following model checkpoints:

1. **AV-HuBERT**: Audio-Visual HuBERT model pretrained on LRS3 dataset
2. **LLM**: A Large Language Model (LLaMA-2, Mistral, etc.)

You can download these models using the provided script:

```bash
# Download AV-HuBERT large model
python scripts/download_models.py --model_name avhubert_large --output_dir checkpoints

# Download LLaMA-2 7B model (requires HuggingFace authentication)
python scripts/download_models.py --model_name llama-2-7b --output_dir checkpoints
```

## Data Preparation

### Full Dataset Preparation

To prepare your dataset for training:

```bash
python scripts/prepare_dataset.py \
    --config configs/default.yaml \
    --input_dir /path/to/videos \
    --output_dir data/processed \
    --transcription_file /path/to/transcriptions.json \
    --val_split 0.05 \
    --test_split 0.05 \
    --workers 4
```

### Demo Dataset Preparation

For testing or demonstration purposes, you can create a small demo dataset:

```bash
python scripts/prepare_demo_dataset.py \
    --source_data /path/to/full/dataset \
    --output_dir data/demo \
    --num_samples 20 \
    --seed 42
```

This will create a small dataset with a few samples for quick testing of the training pipeline.

## Training

### Full Training

To train the AVSR-LLM model:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --checkpoint_dir checkpoints/my_model \
    --data_path data/processed \
    --llm_path checkpoints/llama-2-7b \
    --av_encoder_path checkpoints/avhubert_large.pt \
    --gpu 0
```

Alternatively, you can use the provided shell script:

```bash
bash scripts/train.sh
```

### Test Training with Demo Dataset

To verify the training pipeline works correctly with a small dataset:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --checkpoint_dir checkpoints/test_model \
    --data_path data/demo \
    --llm_path checkpoints/llama-2-7b \
    --av_encoder_path checkpoints/avhubert_large.pt \
    --gpu 0 \
    --debug
```

The `--debug` flag limits training to a small number of steps, which is perfect for testing.

## Memory Management

The codebase includes automatic memory management that will:
1. Detect available GPU memory
2. Calculate model memory requirements
3. Fall back to CPU if the model is too large for the GPU

For best performance:
- Use a GPU with sufficient memory (24GB+ recommended)
- Enable gradient accumulation to reduce memory usage during training
- Consider using 8-bit quantization for the LLM

## Inference

To run inference on audio/video files:

```bash
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/my_model/best_model.pt \
    --input /path/to/video_or_audio \
    --output results \
    --mode av \
    --gpu 0
```

Or use the provided shell script:

```bash
bash scripts/infer_video.sh
```

## Configuration

The model and training parameters can be configured using YAML files. See `configs/default.yaml` for an example configuration.

Key configuration sections:

- `model`: Model architecture and parameters
- `data`: Dataset configuration
- `training`: Training hyperparameters
- `audio`: Audio processing parameters
- `video`: Video processing parameters

## Project Structure

```
.
├── checkpoints/           # Model checkpoints
├── configs/               # Configuration files
├── data/                  # Dataset files
├── examples/              # Example inputs and outputs
├── models/                # Saved model checkpoints
├── scripts/               # Training and inference scripts
├── src/                   # Source code
│   ├── data/              # Data loading and processing
│   ├── inference/         # Inference engine
│   ├── models/            # Model architecture
│   ├── preprocessing/     # Audio/video preprocessing
│   ├── trainer/           # Training logic
│   └── utils/             # Utility functions
├── utils/                 # Additional utilities
├── LICENSE                # License file
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── setup.py               # Package setup file
```

## Supported Models

### Audio Encoders
- Whisper
- HuBERT
- Wav2Vec2

### Visual Encoders
- AV-HuBERT
- ResNet
- EfficientNet

### LLMs
- LLaMA-2
- Mistral
- And other compatible HuggingFace models

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: If you encounter CUDA OOM errors, try:
   - Reduce batch size
   - Enable gradient accumulation (set `gradient_accumulation_steps` > 1)
   - Use CPU fallback (automatic in the current implementation)

2. **fairseq dependency**: The current implementation doesn't require fairseq, but uses a simplified model. To use the exact AV-HuBERT weights, consider adapting the code to your specific fairseq setup.

3. **Dataset compatibility**: Make sure your dataset follows the expected format (TSV manifest files with ID, audio path, and video path columns).

## Citation

If you use this code in your research, please cite:

```
@misc{avsr-llm,
  author = {Your Name},
  title = {AVSR-LLM: Audio-Visual Speech Recognition with Large Language Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/AVSR-LLM}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This code is based on the work from:
- [VSR-LLM](https://github.com/rishabhjain16/VSR-LLM)
- [AV-HuBERT](https://github.com/facebookresearch/av_hubert)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
