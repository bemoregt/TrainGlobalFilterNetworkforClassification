# Global Filter Network (GFNet) for Image Classification

This repository contains an implementation of the Global Filter Network (GFNet) for image classification tasks using PyTorch. GFNet is a vision model that uses global filters in the frequency domain as an alternative to self-attention in vision transformers.

## Overview

GFNet leverages the Fast Fourier Transform (FFT) to efficiently model global context in images by applying learnable filters in the frequency domain. This approach provides several advantages:

- **Global Receptive Field**: Each filter operates on the entire input, capturing global dependencies
- **Efficiency**: Computationally efficient compared to self-attention mechanisms
- **Interpretability**: Frequency domain filters offer enhanced interpretability

## Architecture

The implementation consists of the following components:

- **GlobalFilter**: Applies learnable weights in the frequency domain using FFT
- **GFNetBlock**: Basic building block combining global filtering with MLP
- **GFNet**: Complete network architecture with patch embedding, position embedding, and classification head

## Features

- Optimized for Apple Silicon (MPS) and NVIDIA GPUs (CUDA)
- Automatic device detection and acceleration
- Training and evaluation on CIFAR-10 dataset
- Progress tracking with tqdm
- Visualization of training/validation metrics
- Model checkpointing

## Requirements

```
torch>=1.12.0
torchvision>=0.13.0
matplotlib>=3.5.2
numpy>=1.22.4
tqdm>=4.64.0
```

## Usage

### Training

```bash
python train_gfnet.py
```

The script automatically:
1. Downloads the CIFAR-10 dataset
2. Initializes the GFNet model
3. Trains for 50 epochs with AdamW optimizer and cosine annealing
4. Saves checkpoints every 5 epochs
5. Visualizes training and validation metrics
6. Saves the final model

## Model Configuration

The default configuration is optimized for CIFAR-10:

- Image size: 32×32
- Patch size: 4×4
- Embedding dimension: 256
- Network depth: 8 blocks
- MLP ratio: 4.0
- Batch size: 64

## Performance

On the CIFAR-10 dataset, this implementation of GFNet achieves competitive accuracy while maintaining computational efficiency compared to vision transformers.

## Citation

GFNet is based on the following paper:

```
@article{rao2021global,
  title={Global Filter Networks for Image Classification},
  author={Rao, Yongming and Zhao, Wenliang and Tang, Yansong and Zhou, Jie and Lim, Ser-Nam and Lu, Jiwen},
  journal={arXiv preprint arXiv:2107.00645},
  year={2021}
}
```

## License

MIT