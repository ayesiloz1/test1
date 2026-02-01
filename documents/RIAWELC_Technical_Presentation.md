# RIAWELC - Technical Presentation
## Deep Learning Architecture for Weld Defect Detection

---

# Slide 1: Title

## **RIAWELC**
### Deep Learning-Based Radiographic Image Analysis for Welding Defect Classification

**A Hybrid CNN + CAE Approach**

*Technical Deep Dive*

---

# Slide 2: Agenda

## **Presentation Outline**

1. Problem Statement & Motivation
2. Dataset Description
3. System Architecture Overview
4. CNN Classifier Architecture
5. Convolutional Autoencoder (CAE)
6. GradCAM Explainability
7. Preprocessing Pipeline
8. Training Methodology
9. Evaluation Metrics
10. Results & Performance
11. GUI Implementation
12. Future Work

---

# Slide 3: Problem Statement

## **Technical Challenges in Weld Inspection**

### Traditional NDT Limitations:
- **Manual interpretation** of radiographic images
- **Subjective assessment** leads to inconsistency
- **High cognitive load** causes inspector fatigue
- **No automated traceability**

### Requirements:
- Multi-class defect classification
- Unknown anomaly detection
- Interpretable predictions
- Real-time processing capability

---

# Slide 4: Dataset Overview

## **RIAWELC Dataset Statistics**

### CNN Dataset (4-Class Classification)

| Split | CR | LP | ND | PO | Total |
|-------|-----|-----|------|-----|-------|
| Training | 2,608 | 2,440 | 2,535 | 2,727 | **10,310** |
| Validation | 1,003 | 963 | 975 | 1,024 | **3,965** |
| Testing | 404 | 385 | 390 | 409 | **1,588** |
| **Total** | 4,015 | 3,788 | 3,900 | 4,160 | **15,863** |

### Class Distribution:
- Balanced dataset (±5% variance)
- Stratified splits (65% / 25% / 10%)
- No data leakage between splits

---

# Slide 5: CAE Dataset

## **Unsupervised Learning Dataset**

### Binary Classification (Normal vs Anomaly)

| Split | Normal (ND) | Defect (CR+LP+PO) | Total |
|-------|-------------|-------------------|-------|
| Training | 2,535 | **0** | **2,535** |
| Validation | 975 | 2,990 | **3,965** |
| Testing | 390 | 1,198 | **1,588** |

### Key Design Principle:
- **Training**: Only normal images
- **Validation/Test**: Mixed for threshold optimization
- CAE learns normal distribution → anomalies = high reconstruction error

---

# Slide 6: System Architecture

## **Dual-Model Pipeline**

```
                    ┌─────────────────────────────────┐
                    │         Input Image             │
                    │         (224×224×3)             │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │        Preprocessing          │
                    │   (ImageNet Normalization)    │
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       │                       ▼
    ┌───────────────┐               │               ┌───────────────┐
    │  SimpleCNN    │               │               │     CAE       │
    │  Classifier   │               │               │  Autoencoder  │
    └───────┬───────┘               │               └───────┬───────┘
            │                       │                       │
            ▼                       │                       ▼
    ┌───────────────┐               │               ┌───────────────┐
    │   4-Class     │               │               │  Anomaly      │
    │  Prediction   │               │               │  Score (MSE)  │
    └───────┬───────┘               │               └───────┬───────┘
            │                       │                       │
            ▼                       │                       ▼
    ┌───────────────┐               │               ┌───────────────┐
    │   GradCAM     │               │               │   Heatmap     │
    │   Heatmap     │               │               │  Generation   │
    └───────────────┘               │               └───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │       Combined Results        │
                    │   + Visual Explanations       │
                    └───────────────────────────────┘
```

---

# Slide 7: CNN Architecture - Overview

## **SimpleCNN Classifier**

### Design Philosophy:
- Lightweight architecture (~295K parameters)
- Sufficient capacity for 4-class problem
- Fast inference for real-time applications

### Architecture Summary:
```
Input (224×224×3)
    ↓
[Conv Block 1] → 112×112×32
    ↓
[Conv Block 2] → 56×56×64
    ↓
[Conv Block 3] → 28×28×128  ← GradCAM Target
    ↓
[Global Average Pooling] → 128
    ↓
[FC Layers] → 4 (classes)
```

---

# Slide 8: CNN Architecture - Details

## **Layer-by-Layer Specification**

```python
SimpleCNN(
    features = Sequential(
        # Block 1: 224→112
        Conv2d(3, 32, k=3, p=1),    # [0]
        BatchNorm2d(32),            # [1]
        ReLU(),                     # [2]
        Conv2d(32, 32, k=3, p=1),   # [3]
        BatchNorm2d(32),            # [4]
        ReLU(),                     # [5]
        MaxPool2d(2, 2),            # [6]
        Dropout2d(0.3),             # [7]
        
        # Block 2: 112→56
        Conv2d(32, 64, k=3, p=1),   # [8-15] ...
        
        # Block 3: 56→28
        Conv2d(64, 128, k=3, p=1),  # [16]
        Conv2d(128, 128, k=3, p=1), # [19] ← GradCAM
    ),
    classifier = Sequential(
        Flatten(),
        Linear(128, 64), ReLU(), Dropout(0.3),
        Linear(64, 4)
    )
)
```

---

# Slide 9: CNN - Regularization Techniques

## **Preventing Overfitting**

### 1. **Dropout**
- 2D Dropout (p=0.3) after each conv block
- Standard Dropout (p=0.3) in classifier

### 2. **Batch Normalization**
- After every convolution
- Stabilizes training, enables higher learning rates

### 3. **Data Augmentation**
```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.5)
transforms.RandomRotation(30)
transforms.RandomAffine(translate=(0.3, 0.3))
transforms.ColorJitter(brightness=0.2, contrast=0.2)
```

### 4. **Weight Decay**
- L2 regularization: λ = 1e-4

---

# Slide 10: CAE Architecture

## **Convolutional Autoencoder**

### Encoder-Decoder Structure:

```
ENCODER                              DECODER
─────────────────────────────────────────────────────
Input: 224×224×3
    ↓ ConvBlock(3→32)
224×224×32
    ↓ MaxPool(2×2)
112×112×32
    ↓ ConvBlock(32→64)               ↑ DeconvBlock(64→32)
112×112×64                            112×112×32
    ↓ MaxPool(2×2)                    ↑ Upsample
56×56×64                              56×56×64
    ↓ ConvBlock(64→128)              ↑ DeconvBlock(128→64)
56×56×128                             56×56×128
    ↓ MaxPool(2×2)                    ↑ Upsample
28×28×128                             28×28×128
    ↓ Bottleneck(128→128)            ↑ DeconvBlock(latent→128)
28×28×128 ──────────────────────────→ 28×28×latent

                LATENT SPACE
              (Compressed Representation)
```

---

# Slide 11: CAE - Anomaly Detection Theory

## **How CAE Detects Defects**

### Training Phase:
- Train ONLY on normal images
- CAE learns to reconstruct normal patterns
- Loss function: MSE(input, reconstruction)

### Inference Phase:
```
Normal Image → CAE → Low Reconstruction Error ✓
Defect Image → CAE → High Reconstruction Error ✗
```

### Mathematical Formulation:

$$\text{Anomaly Score} = \frac{1}{C \times H \times W} \sum_{c,h,w} (x_{c,h,w} - \hat{x}_{c,h,w})^2$$

$$\text{Decision} = \begin{cases} \text{Normal} & \text{if score} \leq \theta \\ \text{Anomaly} & \text{if score} > \theta \end{cases}$$

---

# Slide 12: Threshold Optimization

## **Finding Optimal Decision Boundary**

### Method: Maximize F1 Score on Validation Set

```python
def find_optimal_threshold(errors, labels):
    best_f1, best_threshold = 0, 0
    
    for threshold in np.linspace(min(errors), max(errors), 100):
        predictions = (errors > threshold).astype(int)
        f1 = f1_score(labels, predictions)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold
```

### Threshold Selection Trade-offs:
- **Lower threshold** → More sensitive → More false positives
- **Higher threshold** → Less sensitive → More false negatives

---

# Slide 13: GradCAM Theory

## **Gradient-weighted Class Activation Mapping**

### Purpose:
Visual explanation of CNN decisions - WHERE did the model look?

### Mathematical Formulation:

**Step 1**: Compute importance weights (global average pooling of gradients)
$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

**Step 2**: Weighted combination of activation maps
$$L_{GradCAM}^c = ReLU\left(\sum_k \alpha_k^c A^k\right)$$

Where:
- $y^c$ = score for class $c$ (before softmax)
- $A^k$ = activation map of $k$-th channel
- $\alpha_k^c$ = importance weight

---

# Slide 14: GradCAM Implementation

## **Hook-Based Feature Extraction**

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer  # features[19]
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def generate_cam(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Compute CAM
        weights = self.gradients.mean(dim=(2, 3))  # GAP
        cam = (weights @ self.activations).relu()
        cam = normalize(cam)
        
        return cam
```

---

# Slide 15: Preprocessing Pipeline

## **ImageNet Normalization**

### Why ImageNet Stats?
- Pre-trained models expect this normalization
- Consistent with computer vision best practices
- Enables transfer learning compatibility

### Transform Pipeline:

```python
# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize
    transforms.ToTensor(),              # [0,255] → [0,1]
    transforms.Normalize(               # Standardize
        mean=IMAGENET_MEAN, 
        std=IMAGENET_STD
    )
])
```

### Post-processing (for visualization):
```python
def denormalize(tensor):
    return tensor * std + mean  # Back to [0,1]
```

---

# Slide 16: Training Configuration

## **Hyperparameters**

### CNN Training:
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Epochs | 30 |
| LR Scheduler | ReduceLROnPlateau |
| Scheduler Patience | 5 |
| Scheduler Factor | 0.5 |

### CAE Training:
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 1e-5 |
| Batch Size | 64 |
| Epochs | 50 |
| Latent Dimension | 128 |

---

# Slide 17: Loss Functions

## **Training Objectives**

### CNN: Cross-Entropy Loss

$$L_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

```python
criterion = nn.CrossEntropyLoss()
```

### CAE: Mean Squared Error

$$L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$

```python
criterion = nn.MSELoss()
loss = criterion(reconstruction, input_image)
```

---

# Slide 18: Evaluation Metrics

## **Performance Measurement**

### Classification Metrics (CNN):

| Metric | Formula |
|--------|---------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| **Precision** | $\frac{TP}{TP + FP}$ |
| **Recall** | $\frac{TP}{TP + FN}$ |
| **F1 Score** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ |

### Anomaly Detection Metrics (CAE):

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Area under ROC curve |
| **F1 Score** | At optimal threshold |
| **Detection Rate** | Per defect type |

---

# Slide 19: Results - CNN Performance

## **Classification Results**

### Test Set Performance:

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 94.2% |
| **Macro F1 Score** | 0.941 |
| **Weighted F1 Score** | 0.942 |

### Per-Class Performance:

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| CR (Crack) | 0.95 | 0.97 | 0.96 |
| LP (Lack of Penetration) | 0.93 | 0.91 | 0.92 |
| ND (No Defect) | 0.96 | 0.98 | 0.97 |
| PO (Porosity) | 0.93 | 0.91 | 0.92 |

---

# Slide 20: Results - Confusion Matrix

## **CNN Confusion Matrix**

```
              Predicted
           CR   LP   ND   PO
         ┌────┬────┬────┬────┐
    CR   │392 │  5 │  2 │  5 │  → 97.0%
         ├────┼────┼────┼────┤
    LP   │  8 │350 │ 12 │ 15 │  → 90.9%
Actual   ├────┼────┼────┼────┤
    ND   │  1 │  3 │382 │  4 │  → 97.9%
         ├────┼────┼────┼────┤
    PO   │ 11 │ 18 │  7 │373 │  → 91.2%
         └────┴────┴────┴────┘
```

### Observations:
- ND (No Defect) has highest accuracy
- LP and PO show some confusion
- CR detection is very reliable

---

# Slide 21: Results - CAE Performance

## **Anomaly Detection Results**

### Test Set Performance:

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.892 |
| **F1 Score** | 0.847 |
| **Accuracy** | 85.3% |
| **Optimal Threshold** | 0.00142 |

### Per-Defect Detection Rate:

| Defect Type | Detection Rate |
|-------------|----------------|
| CR (Crack) | 91.3% |
| LP (Lack of Penetration) | 82.6% |
| PO (Porosity) | 84.1% |
| ND (Normal) - Correctly Identified | 89.5% |

---

# Slide 22: ROC Curve Analysis

## **CAE Anomaly Detection**

```
    True Positive Rate (Sensitivity)
    1.0 ┤                           ●
        │                       ●
        │                   ●
    0.8 ┤               ●
        │           ●
        │       ●
    0.6 ┤   ●           AUC = 0.892
        │ ●
        │●
    0.4 ┤
        │
        │           ─── ROC Curve
    0.2 ┤           ─ ─ Random (0.5)
        │
        │
    0.0 ┼────┬────┬────┬────┬────┬
        0   0.2  0.4  0.6  0.8  1.0
              False Positive Rate
```

---

# Slide 23: Reconstruction Error Distribution

## **Normal vs Defect Separation**

```
    Frequency
    │
    │    ┌───┐
    │    │   │
    │   ┌┤   │
    │   ││   │          ┌───┐
    │  ┌┤│   │         ┌┤   │
    │  │││   │        ┌┤│   │
    │ ┌┤││   │       ┌┤││   │
    │ ││││   │      ┌┤│││   │
    │┌┤│││   │     ┌┤││││   │
    └┴┴┴┴┴───┴─────┴┴┴┴┴┴───┴────→ Error
         │              │
       Normal        Defect
       (Low MSE)    (High MSE)
                 ↑
             Threshold
```

---

# Slide 24: Inference Speed

## **Performance Benchmarks**

### Single Image Inference:

| Hardware | CNN Only | CNN + GradCAM | CNN + CAE |
|----------|----------|---------------|-----------|
| CPU (i7-10700) | 45 ms | 120 ms | 180 ms |
| GPU (RTX 3070) | 3 ms | 8 ms | 12 ms |
| GPU (T4 Colab) | 5 ms | 12 ms | 18 ms |

### Batch Processing (100 images):

| Hardware | Time | Throughput |
|----------|------|------------|
| CPU | 18 sec | 5.5 img/sec |
| GPU | 1.2 sec | 83 img/sec |

---

# Slide 25: GUI Architecture

## **PyQt5 Application Structure**

```
WeldDefectGUI (QMainWindow)
├── Tab Widget
│   ├── Tab 1: Input Image
│   ├── Tab 2: CNN GradCAM
│   ├── Tab 3: AE Reconstruction
│   ├── Tab 4: AE Anomaly Map
│   └── Tab 5: Results
├── Control Panel
│   ├── Load Image / Load Folder
│   ├── Analyze / Analyze All
│   └── Navigation (Prev / Next)
├── Settings Panel
│   ├── Use CNN Checkbox
│   ├── Use Autoencoder Checkbox
│   └── AE Threshold Spinner
└── Inference Engine
    ├── SimpleCNN Model
    ├── CAE Model
    └── GradCAM Generator
```

---

# Slide 26: Key Implementation Features

## **Technical Highlights**

### 1. **Batch Processing with Caching**
```python
self.batch_results_cache = {}  # Path → Result mapping
```

### 2. **Ground Truth Extraction**
```python
# From folder structure: .../CR/image.png → "CR"
ground_truth = Path(image_path).parent.name
```

### 3. **Thread-Safe GUI Updates**
```python
QApplication.processEvents()  # Keep UI responsive
```

### 4. **Memory Management**
```python
with torch.no_grad():  # Disable gradient computation
    result = model(input)
```

---

# Slide 27: Data Leakage Prevention

## **Ensuring Valid Evaluation**

### Problem:
Random splitting can cause same/similar images in train and test

### Solution:
1. **Preserve original splits** from source dataset
2. **Verification script** to check overlap

```python
def check_leakage(train_files, test_files):
    train_set = set(f.name for f in train_files)
    test_set = set(f.name for f in test_files)
    overlap = train_set & test_set
    
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} files")
    print("✓ No data leakage detected")
```

### Verified Results:
- CNN Dataset: 0 overlapping files
- CAE Dataset: 0 overlapping files

---

# Slide 28: Model Serialization

## **Checkpoint Format**

### CNN Model:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'accuracy': best_accuracy,
    'config': training_config
}, 'best_model_pytorch.pth')
```

### CAE Model:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'threshold': optimal_threshold,
    'auc': test_auc,
    'f1': test_f1,
    'config': {
        'latent_dim': 128,
        'image_size': 224
    }
}, 'cae_final.pth')
```

---

# Slide 29: Limitations & Challenges

## **Known Limitations**

### 1. **Dataset Bias**
- Trained on specific X-ray equipment
- May need fine-tuning for different setups

### 2. **CAE Threshold Sensitivity**
- Optimal threshold varies with data distribution
- Requires calibration for new deployments

### 3. **Similar Defect Confusion**
- LP and PO can appear similar in some cases
- May benefit from attention mechanisms

### 4. **Computational Requirements**
- GPU recommended for real-time batch processing
- CPU inference acceptable for single images

---

# Slide 30: Future Work

## **Planned Improvements**

### Short Term:
- [ ] Attention mechanism integration (CBAM, SE blocks)
- [ ] Multi-scale feature fusion
- [ ] Model quantization for edge deployment

### Medium Term:
- [ ] Vision Transformer (ViT) comparison
- [ ] Semi-supervised learning with limited labels
- [ ] Active learning for efficient labeling

### Long Term:
- [ ] 3D volumetric analysis (CT data)
- [ ] Real-time video stream processing
- [ ] Federated learning for multi-site deployment

---

# Slide 31: Reproducibility

## **Code & Environment**

### Requirements:
```
torch>=2.0.0
torchvision>=0.15.0
PyQt5>=5.15.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

### Training Commands:
```bash
# CNN Training
python train_pytorch.py --epochs 30 --batch_size 32

# CAE Training
cd cae/src
python train.py --epochs 50 --batch_size 64 --latent_dim 128

# Or on Google Colab (GPU)
# Upload and run train_colab.ipynb
```

---

# Slide 32: Summary

## **Key Takeaways**

### 1. **Dual-Model Approach**
- CNN for multi-class classification
- CAE for unknown anomaly detection

### 2. **Explainability**
- GradCAM provides visual explanations
- Builds trust in AI decisions

### 3. **Performance**
- CNN: 94.2% accuracy
- CAE: 0.892 AUC-ROC
- Real-time inference capability

### 4. **Practical Deployment**
- User-friendly GUI
- Batch processing support
- Ground truth comparison

---

# Slide 33: References

## **Key Publications**

1. Selvaraju, R.R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.

2. An, J., & Cho, S. (2015). "Variational Autoencoder based Anomaly Detection using Reconstruction Probability." SNU Data Mining Center.

3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

4. Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training." ICML.

5. ASME BPVC Section V - Nondestructive Examination (2021).

---

# Slide 34: Questions?

## **Thank You**

### Contact Information:
- **Project Repository**: github.com/user/RIAWELC
- **Documentation**: See `Documents/` folder

### Demo Available:
```bash
cd gui
python main.py
```

---

*Technical Presentation v1.0 - February 2026*
