# Convolutional Autoencoder (CAE) for Weld Defect Detection

This folder contains the implementation of a Convolutional Autoencoder for **unsupervised anomaly detection** in weld images.

## Overview

The CAE is trained **only on normal (ND) images** to learn the representation of defect-free welds. At inference time, images with defects will have high reconstruction error since the model never learned to reconstruct them.

## Directory Structure

```
cae/
├── README.md                 # This file
├── create_dataset.py         # Script to create CAE-specific dataset
├── dataset/                  # Generated dataset (after running create_dataset.py)
│   ├── training/
│   │   └── normal/          # Only ND images (80%)
│   ├── validation/
│   │   ├── normal/          # ND images (10%)
│   │   └── defect/
│   │       ├── CR/
│   │       ├── LP/
│   │       └── PO/
│   └── testing/
│       ├── normal/          # ND images (10%)
│       └── defect/
│           ├── CR/
│           ├── LP/
│           └── PO/
├── src/
│   ├── __init__.py
│   ├── model.py             # CAE architecture
│   ├── dataset.py           # PyTorch datasets and dataloaders
│   └── train.py             # Training script
└── models/                   # Saved models (after training)
    ├── best_model.pth
    ├── final_model.pth
    ├── training_history.png
    ├── roc_curve.png
    └── error_distribution.png
```

## Quick Start

### 1. Create Dataset

```bash
cd cae
python create_dataset.py
```

This will:
- Copy ND images as "normal" (80% train, 10% val, 10% test)
- Copy CR, LP, PO images as "defects" (30% val, 70% test)

### 2. Train the Model

```bash
cd src
python train.py --epochs 100 --batch_size 32
```

Options:
- `--model_type`: `standard` (default) or `large`
- `--latent_dim`: Latent space dimension (default: 128)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-3)

### 3. Test the Model

```bash
python train.py --test_only --model_path ../models/best_model.pth
```

## Model Architecture

### Standard CAE
- **Encoder**: 3 conv blocks with max pooling (224→112→56→28)
- **Bottleneck**: 128-dim feature map at 28x28
- **Decoder**: 3 deconv blocks (28→56→112→224)
- **Output**: Sigmoid activation (values in [0, 1])
- **Parameters**: ~1.5M

### Large CAE
- **Encoder**: 5 conv blocks with strided convolutions
- **Bottleneck**: 256-dim feature map at 7x7
- **Decoder**: 5 deconv blocks
- **Output**: Sigmoid activation
- **Parameters**: ~5M

## Training Strategy

1. **Train only on normal images**: The autoencoder learns to reconstruct defect-free welds
2. **Validation**: Compute reconstruction error on both normal and defect images
3. **Threshold tuning**: Find optimal threshold using F1 score on validation set
4. **Testing**: Evaluate final performance with the learned threshold

## Anomaly Detection

At inference:
1. Compute reconstruction error: `MSE(input, reconstruction)`
2. If `error > threshold`: **DEFECT** (anomaly)
3. If `error ≤ threshold`: **NORMAL**

## Integration with GUI

The trained model can be loaded in the GUI's inference engine:

```python
from cae.src.model import CAE

# Load model
model = CAE()
model.load_state_dict(torch.load('cae/models/best_model.pth')['model_state_dict'])
model.eval()

# Get reconstruction error
error, reconstruction = model.get_reconstruction_error(image_tensor)

# Anomaly detection
is_anomaly = error > threshold
```

## Expected Results

Based on similar datasets, you can expect:
- **AUC**: 0.85 - 0.95
- **Recall (defect detection)**: 85% - 95%
- **Precision**: 70% - 85%

The exact performance depends on:
- Dataset size and quality
- Difference between normal and defect images
- Model capacity and training duration
