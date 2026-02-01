# RIAWELC - Weld Defect Detection System
## Complete Project Documentation

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset Structure](#3-dataset-structure)
4. [Models](#4-models)
5. [GUI Application](#5-gui-application)
6. [Installation & Setup](#6-installation--setup)
7. [Usage Guide](#7-usage-guide)
8. [Technical Details](#8-technical-details)

---

## 1. Project Overview

### What is RIAWELC?
RIAWELC (Radiographic Image Analysis for Welding Defect Classification) is an AI-powered system for detecting and classifying defects in radiographic images of welds. The system uses deep learning to automatically identify four types of weld conditions:

| Class | Code | Description |
|-------|------|-------------|
| **Crack** | CR | Linear discontinuities in the weld |
| **Lack of Penetration** | LP | Incomplete fusion at the weld root |
| **No Defect** | ND | Normal, defect-free weld |
| **Porosity** | PO | Gas pockets/voids in the weld |

### Key Features
- ✅ **Supervised Classification (CNN)**: 4-class defect classification with high accuracy
- ✅ **GradCAM Visualization**: Visual explanation of model decisions
- ✅ **Unsupervised Anomaly Detection (CAE)**: Detect unknown defects
- ✅ **Batch Processing**: Analyze entire folders of images
- ✅ **Ground Truth Comparison**: Compare predictions with actual labels
- ✅ **User-Friendly GUI**: Dark-themed PyQt5 interface

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RIAWELC System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Input      │    │   Models     │    │   Output     │      │
│  │   Images     │───▶│              │───▶│   Results    │      │
│  │  (X-ray)     │    │  CNN + CAE   │    │  + Visuals   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GUI Application                       │   │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┐   │   │
│  │  │  Input  │ GradCAM │   AE    │ Anomaly │ Results │   │   │
│  │  │  Tab    │   Tab   │  Recon  │   Map   │   Tab   │   │   │
│  │  └─────────┴─────────┴─────────┴─────────┴─────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure
```
RIAWELC/
├── gui/                          # GUI Application
│   ├── main.py                   # Main GUI window (PyQt5)
│   ├── inference_engine.py       # Model inference + GradCAM
│   ├── config.py                 # Configuration settings
│   └── gui_outputs/              # Saved analysis results
│
├── models/                       # Trained Models
│   ├── best_model_pytorch.pth    # CNN classifier (best)
│   └── weld_defect_pytorch.pth   # CNN classifier
│
├── dataset/                      # CNN Dataset (4 classes)
│   ├── training/                 # 10,310 images
│   │   ├── CR/, LP/, ND/, PO/
│   ├── validation/               # 3,965 images
│   │   ├── CR/, LP/, ND/, PO/
│   └── testing/                  # 1,588 images
│       ├── CR/, LP/, ND/, PO/
│
├── cae/                          # Convolutional Autoencoder
│   ├── src/
│   │   ├── model.py              # CAE architecture
│   │   ├── dataset.py            # Data loading
│   │   └── train.py              # Training script
│   ├── dataset/                  # CAE Dataset (binary)
│   │   ├── training/normal/      # Normal images only
│   │   ├── validation/           # Normal + Defects
│   │   └── testing/              # Normal + Defects
│   ├── train_colab.ipynb         # Google Colab training
│   └── models/                   # Trained CAE models
│
├── train_pytorch.py              # CNN training script
├── gradcam_visualization.py      # GradCAM utilities
└── data_preprocessing.ipynb      # Data preparation
```

---

## 3. Dataset Structure

### CNN Dataset (Supervised - 4 Classes)
Used for training the SimpleCNN classifier.

| Split | CR | LP | ND | PO | Total |
|-------|-----|-----|------|-----|-------|
| Training | 2,608 | 2,440 | 2,535 | 2,727 | **10,310** |
| Validation | 1,003 | 963 | 975 | 1,024 | **3,965** |
| Testing | 404 | 385 | 390 | 409 | **1,588** |
| **Total** | 4,015 | 3,788 | 3,900 | 4,160 | **15,863** |

### CAE Dataset (Unsupervised - Binary)
Used for training the Convolutional Autoencoder (anomaly detection).

| Split | Normal (ND) | Defect (CR+LP+PO) | Total |
|-------|-------------|-------------------|-------|
| Training | 2,535 | 0 | **2,535** |
| Validation | 975 | 2,990 | **3,965** |
| Testing | 390 | 1,198 | **1,588** |

**Key Principle**: CAE is trained ONLY on normal images. During inference, defects produce higher reconstruction error (anomaly score).

---

## 4. Models

### 4.1 SimpleCNN (Supervised Classifier)

**Purpose**: Classify images into 4 defect categories (CR, LP, ND, PO)

**Architecture**:
```
Input: 224×224×3 RGB Image
    ↓
┌─────────────────────────────────────┐
│ Block 1: Conv(3→32) + BN + ReLU     │
│          Conv(32→32) + BN + ReLU    │
│          MaxPool(2×2) + Dropout     │
└─────────────────────────────────────┘
    ↓ 112×112×32
┌─────────────────────────────────────┐
│ Block 2: Conv(32→64) + BN + ReLU    │
│          Conv(64→64) + BN + ReLU    │
│          MaxPool(2×2) + Dropout     │
└─────────────────────────────────────┘
    ↓ 56×56×64
┌─────────────────────────────────────┐
│ Block 3: Conv(64→128) + BN + ReLU   │
│          Conv(128→128) + BN + ReLU  │ ← GradCAM Target Layer
│          MaxPool(2×2) + Dropout     │
└─────────────────────────────────────┘
    ↓ 28×28×128
┌─────────────────────────────────────┐
│ Global Average Pooling              │
│ FC(128→64) + ReLU + Dropout         │
│ FC(64→4) → Softmax                  │
└─────────────────────────────────────┘
    ↓
Output: [P(CR), P(LP), P(ND), P(PO)]
```

**Parameters**: ~200K trainable parameters

**Preprocessing** (ImageNet Normalization):
```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
```

---

### 4.2 Convolutional Autoencoder (CAE)

**Purpose**: Detect anomalies by measuring reconstruction error

**How It Works**:
1. Train only on NORMAL images
2. CAE learns to reconstruct normal patterns
3. Defective images → High reconstruction error
4. Threshold determines normal vs anomaly

**Architecture**:
```
Input: 224×224×3 RGB Image
    ↓
┌─────────────────────────────────────┐
│ ENCODER                             │
│ Conv(3→32) + Pool → 112×112×32      │
│ Conv(32→64) + Pool → 56×56×64       │
│ Conv(64→128) + Pool → 28×28×128     │
│ Bottleneck → 28×28×128              │
└─────────────────────────────────────┘
    ↓ Latent Space (compressed)
┌─────────────────────────────────────┐
│ DECODER                             │
│ DeConv(128→128) → 56×56×128         │
│ DeConv(128→64) → 112×112×64         │
│ DeConv(64→32) → 224×224×32          │
│ Conv(32→3) → 224×224×3              │
└─────────────────────────────────────┘
    ↓
Output: Reconstructed Image

Anomaly Score = MSE(Input, Output)
If score > threshold → ANOMALY (Defect)
```

**Parameters**: ~886K trainable parameters

**Preprocessing**: Same as CNN (ImageNet normalization)

---

### 4.3 GradCAM (Visualization)

**Purpose**: Explain WHY the CNN made its prediction

**How It Works**:
1. Forward pass through CNN
2. Compute gradients of predicted class w.r.t. last conv layer
3. Weight activation maps by gradient importance
4. Generate heatmap showing important regions

```
Original Image + GradCAM Heatmap = Visual Explanation
     ↓                  ↓                   ↓
  [X-ray]    +    [Red=Important]    =   [Overlay]
```

**Target Layer**: `features[19]` (last Conv2d in Block 3)

---

## 5. GUI Application

### 5.1 Main Window Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  RIAWELC - Weld Defect Detection System                    [_][□][X]│
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [Input Image] │ [CNN GradCAM] │ [AE Recon] │ [Anomaly] │ [Results]   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │                                                         │   │
│  │              IMAGE DISPLAY AREA                         │   │
│  │                                                         │   │
│  │                                                         │   │
│  │  Ground Truth: [ND]                                     │   │
│  │  Info: Description of current tab                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ [Load Image] [Load Folder] [Analyze] [Analyze All]        │ │
│  │ [Prev] [1/100] [Next]                    [Clear All]      │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Analysis Settings                                         │ │
│  │ ☑ Use CNN Classification                                  │ │
│  │ ☐ Use Autoencoder Anomaly Detection                       │ │
│  │ ☐ Apply Gaussian Blur                                     │ │
│  │ AE Threshold: [0.001]                                     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Tab Descriptions

| Tab | Purpose | Display |
|-----|---------|---------|
| **Input Image** | Original X-ray image | Raw image + Ground Truth label |
| **CNN GradCAM** | CNN classification + explanation | Heatmap overlay showing important regions |
| **AE Reconstruction** | Autoencoder output | Side-by-side: Original vs Reconstructed |
| **AE Anomaly Map** | Pixel-wise error visualization | Heatmap of reconstruction differences |
| **Results** | Combined analysis summary | All predictions + confidence scores |

### 5.3 Key Features

#### Single Image Analysis
1. Click **"Load Image"**
2. Select an X-ray image
3. Click **"Analyze"**
4. View results across all tabs

#### Batch Processing
1. Click **"Load Folder"**
2. Select folder containing subfolders: `CR/`, `LP/`, `ND/`, `PO/`
3. Click **"Analyze All"**
4. Navigate with **"Prev"** / **"Next"** buttons
5. Results are cached for quick navigation

#### Ground Truth Display
- Automatically extracted from folder structure
- Shown on ALL tabs for easy comparison
- Format: `Ground Truth: [CLASS_NAME]`

---

## 6. Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Step 1: Create Environment
```bash
conda create -n steel_defect_detection python=3.10
conda activate steel_defect_detection
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision
pip install PyQt5
pip install opencv-python
pip install numpy matplotlib scikit-learn
pip install tqdm pillow
```

### Step 3: Verify Models
Ensure these files exist:
```
models/best_model_pytorch.pth    # CNN model
cae/models/cae_final.pth         # CAE model (after training)
```

### Step 4: Run GUI
```bash
cd gui
python main.py
```

---

## 7. Usage Guide

### 7.1 Basic Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Load Image  │────▶│   Analyze    │────▶│ View Results │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 7.2 Interpreting Results

#### CNN Classification
- **Class**: Predicted defect type (CR, LP, ND, PO)
- **Confidence**: Model certainty (0-100%)
- **GradCAM**: Red regions = important for decision

#### Autoencoder Anomaly Detection
- **Reconstruction Error**: Lower = more normal
- **Threshold**: Decision boundary
- **Anomaly Score > Threshold** → Defect detected

### 7.3 Best Practices

1. **Use both models together** for robust detection
2. **Check GradCAM** to understand model reasoning
3. **Adjust AE threshold** based on your requirements:
   - Lower threshold = More sensitive (more false positives)
   - Higher threshold = Less sensitive (may miss defects)

---

## 8. Technical Details

### 8.1 Preprocessing Pipeline

Both CNN and CAE use identical preprocessing:

```python
# ImageNet Normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
```

### 8.2 Model Files

#### CNN Model (`best_model_pytorch.pth`)
```python
{
    'model_state_dict': state_dict,
    'optimizer_state_dict': optimizer_state,
    'epoch': int,
    'accuracy': float
}
```

#### CAE Model (`cae_final.pth`)
```python
{
    'model_state_dict': state_dict,
    'threshold': float,        # Optimal anomaly threshold
    'test_auc': float,         # AUC score on test set
    'test_f1': float,          # F1 score on test set
    'config': dict             # Training configuration
}
```

### 8.3 Key Files Reference

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `gui/main.py` | Main GUI application | `WeldDefectGUI` |
| `gui/inference_engine.py` | Model inference | `DefectDetectionEngine`, `GradCAM` |
| `gui/config.py` | Configuration | `GUIConfig` |
| `train_pytorch.py` | CNN training | `SimpleCNN`, `Config` |
| `cae/src/model.py` | CAE architecture | `CAE`, `CAELarge` |
| `cae/src/dataset.py` | Data loading | `NormalDataset`, `AnomalyDataset` |
| `cae/src/train.py` | CAE training | `CAETrainer` |

### 8.4 Performance Metrics

| Model | Metric | Expected Value |
|-------|--------|----------------|
| CNN | Accuracy | 85-95% |
| CNN | F1 Score | 0.85-0.95 |
| CAE | AUC Score | 0.80-0.95 |
| CAE | F1 Score | 0.75-0.90 |

---

## Quick Reference Card

### Keyboard Shortcuts (GUI)
- `Ctrl+O`: Load Image
- `Ctrl+F`: Load Folder
- `Enter`: Analyze
- `Left Arrow`: Previous Image
- `Right Arrow`: Next Image

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Model not found" | Check paths in `config.py` |
| "CUDA out of memory" | Reduce batch size or use CPU |
| "No images found" | Ensure folder structure: `CR/`, `LP/`, `ND/`, `PO/` |
| Slow inference | Enable GPU or use smaller image size |

---

## Contact & Resources

- **Project**: RIAWELC - Weld Defect Detection
- **Models**: SimpleCNN + Convolutional Autoencoder
- **Framework**: PyTorch + PyQt5
- **Dataset**: 15,863 radiographic images

---

*Documentation generated: February 2026*
