# RIAWELC - Technical Reference Manual
## Deep Learning Weld Defect Detection System

---

## Table of Contents
1. [System Requirements & Dependencies](#1-system-requirements--dependencies)
2. [Model Architectures](#2-model-architectures)
3. [Data Pipeline](#3-data-pipeline)
4. [Inference Engine API](#4-inference-engine-api)
5. [GradCAM Implementation](#5-gradcam-implementation)
6. [Loss Functions & Metrics](#6-loss-functions--metrics)
7. [Training Procedures](#7-training-procedures)
8. [GUI Implementation](#8-gui-implementation)
9. [Configuration Reference](#9-configuration-reference)
10. [Troubleshooting & Debugging](#10-troubleshooting--debugging)

---

## 1. System Requirements & Dependencies

### 1.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | - | NVIDIA GTX 1060+ (6GB VRAM) |
| Storage | 5 GB | 10+ GB (for datasets) |

### 1.2 Software Dependencies

```
# Core ML Framework
torch>=2.0.0
torchvision>=0.15.0

# GUI Framework
PyQt5>=5.15.0

# Image Processing
opencv-python>=4.8.0
Pillow>=9.0.0

# Scientific Computing
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0

# Utilities
tqdm>=4.65.0
```

### 1.3 CUDA Configuration

```python
# Check CUDA availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
```

---

## 2. Model Architectures

### 2.1 SimpleCNN Architecture

**File**: `train_pytorch.py`

```python
class SimpleCNN(nn.Module):
    """
    Simple CNN for 4-class weld defect classification.
    
    Architecture Summary:
    - 3 Convolutional blocks with increasing channels (32→64→128)
    - BatchNorm + ReLU activation after each conv layer
    - MaxPooling (2x2) with Dropout for regularization
    - Global Average Pooling → FC layers → 4-class output
    
    Input: (B, 3, 224, 224) - Batch of RGB images
    Output: (B, 4) - Class logits [CR, LP, ND, PO]
    """
    
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 224x224x3 → 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),      # [0]
            nn.BatchNorm2d(32),                               # [1]
            nn.ReLU(inplace=True),                            # [2]
            nn.Conv2d(32, 32, kernel_size=3, padding=1),     # [3]
            nn.BatchNorm2d(32),                               # [4]
            nn.ReLU(inplace=True),                            # [5]
            nn.MaxPool2d(2, 2),                               # [6]
            nn.Dropout2d(dropout_rate),                       # [7]
            
            # Block 2: 112x112x32 → 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # [8]
            nn.BatchNorm2d(64),                               # [9]
            nn.ReLU(inplace=True),                            # [10]
            nn.Conv2d(64, 64, kernel_size=3, padding=1),     # [11]
            nn.BatchNorm2d(64),                               # [12]
            nn.ReLU(inplace=True),                            # [13]
            nn.MaxPool2d(2, 2),                               # [14]
            nn.Dropout2d(dropout_rate),                       # [15]
            
            # Block 3: 56x56x64 → 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # [16]
            nn.BatchNorm2d(128),                              # [17]
            nn.ReLU(inplace=True),                            # [18]
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   # [19] ← GradCAM Target
            nn.BatchNorm2d(128),                              # [20]
            nn.ReLU(inplace=True),                            # [21]
            nn.MaxPool2d(2, 2),                               # [22]
            nn.Dropout2d(dropout_rate),                       # [23]
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 28x28x128 → 1x1x128
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
```

**Parameter Count**:
```
Block 1: (3×3×3×32 + 32) + (3×3×32×32 + 32) = 9,344
Block 2: (3×3×32×64 + 64) + (3×3×64×64 + 64) = 55,552
Block 3: (3×3×64×128 + 128) + (3×3×128×128 + 128) = 221,568
Classifier: (128×64 + 64) + (64×4 + 4) = 8,516
Total: ~295,000 parameters
```

### 2.2 CAE Architecture

**File**: `cae/src/model.py`

```python
class CAE(nn.Module):
    """
    Convolutional Autoencoder for Anomaly Detection.
    
    Training: Only on normal (ND) images
    Inference: High reconstruction error → Anomaly (defect)
    
    Encoder: 224×224×3 → 28×28×latent_dim
    Decoder: 28×28×latent_dim → 224×224×3
    
    Compression ratio: 224×224×3 / 28×28×128 = 1.5x
    """
    
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)    # 224 → 224
        self.pool1 = nn.MaxPool2d(2, 2)           # 224 → 112
        self.enc2 = ConvBlock(32, 64)             # 112 → 112
        self.pool2 = nn.MaxPool2d(2, 2)           # 112 → 56
        self.enc3 = ConvBlock(64, 128)            # 56 → 56
        self.pool3 = nn.MaxPool2d(2, 2)           # 56 → 28
        self.bottleneck = ConvBlock(128, latent_dim)  # 28 → 28
        
        # Decoder
        self.dec3 = DeconvBlock(latent_dim, 128)  # 28 → 56
        self.dec2 = DeconvBlock(128, 64)          # 56 → 112
        self.dec1 = DeconvBlock(64, 32)           # 112 → 224
        
        # Output (no activation - ImageNet normalized output)
        self.output = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
    
    def encode(self, x):
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.bottleneck(x)
        return x
    
    def decode(self, z):
        x = self.dec3(z)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.output(x)
        return x
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def get_reconstruction_error(self, x, reduction='mean'):
        """
        Calculate MSE reconstruction error.
        
        Args:
            x: Input tensor (B, C, H, W)
            reduction: 'mean' → scalar, 'none' → pixel-wise
        
        Returns:
            error: Reconstruction error
            reconstruction: Reconstructed image
        """
        with torch.no_grad():
            reconstruction = self.forward(x)
            
            if reduction == 'mean':
                # Scalar error per sample: mean over C, H, W
                error = torch.mean((x - reconstruction) ** 2, dim=[1, 2, 3])
            elif reduction == 'none':
                # Pixel-wise error for heatmap
                error = torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)
            
        return error, reconstruction
```

**ConvBlock & DeconvBlock**:
```python
class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class DeconvBlock(nn.Module):
    """ConvTranspose (upsample) → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
```

**Parameter Count**:
```
Encoder:
  enc1: 2×(3×3×3×32 + 3×3×32×32) + BN = 10,400
  enc2: 2×(3×3×32×64 + 3×3×64×64) + BN = 55,680
  enc3: 2×(3×3×64×128 + 3×3×128×128) + BN = 222,080
  bottleneck: 2×(3×3×128×128) + BN = 295,680

Decoder:
  dec3: ConvT(128→128) + Conv = 295,936
  dec2: ConvT(128→64) + Conv = 110,720
  dec1: ConvT(64→32) + Conv = 27,744
  output: 3×3×32×3 = 867

Total: ~886,000 parameters
```

---

## 3. Data Pipeline

### 3.1 Preprocessing (ImageNet Normalization)

```python
# Constants (matching ImageNet pre-training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training Transform (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),  # [0, 255] → [0, 1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Validation/Test Transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
```

### 3.2 Denormalization (for visualization)

```python
def denormalize(tensor):
    """
    Convert ImageNet-normalized tensor back to [0, 1] for display.
    
    Formula: x_original = x_normalized * std + mean
    
    Args:
        tensor: (C, H, W) or (B, C, H, W) normalized tensor
    
    Returns:
        Denormalized tensor clamped to [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    
    if tensor.device.type != 'cpu':
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    if tensor.dim() == 4:  # Batch dimension
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)
```

### 3.3 Dataset Classes

**NormalDataset** (CAE Training):
```python
class NormalDataset(Dataset):
    """
    Loads only NORMAL images for unsupervised CAE training.
    
    Returns: (image_tensor, path_string)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Supported extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.root_dir.glob(ext))
        
        self.image_paths = sorted(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(self.image_paths[idx])
```

**AnomalyDataset** (CAE Validation/Testing):
```python
class AnomalyDataset(Dataset):
    """
    Loads normal + defect images with binary labels.
    
    Labels: 0 = Normal (ND), 1 = Defect (CR, LP, PO)
    Returns: (image_tensor, label, defect_type, path_string)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.defect_types = []
        
        # Load normal (label=0)
        normal_dir = self.root_dir / 'normal'
        for img in normal_dir.glob('*.[pjbt][npmi][gfp]*'):
            self.image_paths.append(img)
            self.labels.append(0)
            self.defect_types.append('ND')
        
        # Load defects (label=1)
        defect_dir = self.root_dir / 'defect'
        for defect_type in ['CR', 'LP', 'PO']:
            type_dir = defect_dir / defect_type
            if type_dir.exists():
                for img in type_dir.glob('*.[pjbt][npmi][gfp]*'):
                    self.image_paths.append(img)
                    self.labels.append(1)
                    self.defect_types.append(defect_type)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.defect_types[idx], str(self.image_paths[idx])
```

### 3.4 DataLoader Configuration

```python
def create_dataloaders(dataset_dir, batch_size=32, image_size=224, 
                       num_workers=4, pin_memory=True):
    """
    Create train/val/test DataLoaders.
    
    Args:
        dataset_dir: Path to CAE dataset
        batch_size: Samples per batch
        image_size: Target image size
        num_workers: Parallel data loading workers
        pin_memory: Pin to GPU memory (set False for CPU)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_dir = Path(dataset_dir)
    
    # Transforms
    train_transform = get_transforms(image_size, augment=True)
    val_transform = get_transforms(image_size, augment=False)
    
    # Datasets
    train_dataset = NormalDataset(
        dataset_dir / 'training' / 'normal', 
        train_transform
    )
    val_dataset = AnomalyDataset(
        dataset_dir / 'validation', 
        val_transform
    )
    test_dataset = AnomalyDataset(
        dataset_dir / 'testing', 
        val_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
```

---

## 4. Inference Engine API

**File**: `gui/inference_engine.py`

### 4.1 DefectDetectionEngine Class

```python
class DefectDetectionEngine:
    """
    Combined inference engine for CNN classification + CAE anomaly detection.
    
    Attributes:
        cnn_model: SimpleCNN classifier
        autoencoder_model: CAE for anomaly detection
        gradcam: GradCAM visualizer
        class_names: ['CR', 'LP', 'ND', 'PO']
        device: torch.device (cuda/cpu)
    """
    
    def __init__(self, config):
        """
        Initialize engine with configuration.
        
        Args:
            config: GUIConfig object with model paths
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_cnn()       # Load SimpleCNN
        self._load_autoencoder()  # Load CAE (optional)
        
        self.class_names = ['CR', 'LP', 'ND', 'PO']
```

### 4.2 Prediction API

```python
def predict(self, image_path, config=None):
    """
    Run full prediction pipeline on single image.
    
    Args:
        image_path: Path to input image
        config: Optional config override
    
    Returns:
        dict: {
            'cnn': {
                'class_name': str,       # 'CR', 'LP', 'ND', 'PO'
                'class_idx': int,        # 0, 1, 2, 3
                'confidence': float,     # 0.0 - 1.0
                'probabilities': list,   # [p_CR, p_LP, p_ND, p_PO]
                'gradcam_heatmap': np.ndarray,  # (H, W) float32
                'gradcam_overlay': np.ndarray   # (H, W, 3) uint8 RGB
            },
            'autoencoder': {
                'error': float,          # Reconstruction MSE
                'threshold': float,      # Decision threshold
                'is_anomaly': bool,      # error > threshold
                'reconstruction': torch.Tensor,  # (3, H, W)
                'heatmap': torch.Tensor          # (1, H, W)
            },
            'processing_time': float  # Seconds
        }
    """
```

### 4.3 CNN Prediction

```python
def _predict_cnn(self, image_tensor, original_image=None):
    """
    CNN classification with GradCAM visualization.
    
    Args:
        image_tensor: (1, 3, 224, 224) normalized tensor
        original_image: (224, 224, 3) numpy array for overlay
    
    Returns:
        dict with classification results and GradCAM
    """
    # Forward pass
    with torch.no_grad():
        outputs = self.cnn_model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
    
    # GradCAM (requires gradients)
    gradcam_heatmap, _, _ = self.gradcam.generate_cam(
        image_tensor, 
        target_class=predicted.item()
    )
    
    # Create overlay
    gradcam_overlay = self._create_gradcam_overlay(
        original_image, 
        gradcam_heatmap
    )
    
    return {
        'class_name': self.class_names[predicted.item()],
        'class_idx': predicted.item(),
        'confidence': confidence.item(),
        'probabilities': probabilities[0].cpu().numpy().tolist(),
        'gradcam_heatmap': gradcam_heatmap,
        'gradcam_overlay': gradcam_overlay
    }
```

### 4.4 Autoencoder Prediction

```python
def _predict_autoencoder(self, image_tensor):
    """
    Autoencoder anomaly detection.
    
    Args:
        image_tensor: (1, 3, 224, 224) normalized tensor
    
    Returns:
        dict with reconstruction error and anomaly classification
    """
    with torch.no_grad():
        # Get reconstruction and error
        error, reconstruction = self.autoencoder_model.get_reconstruction_error(
            image_tensor, reduction='none'
        )
        
        error_value = error.mean().item()  # Scalar MSE
        threshold = self.config.ae_threshold or self.ae_threshold
        is_anomaly = error_value > threshold
        
        # Pixel-wise error heatmap
        diff = torch.abs(image_tensor - reconstruction)
        diff_normalized = diff / (diff.max() + 1e-8)
        heatmap = diff_normalized.mean(dim=1, keepdim=True)
        
        return {
            'error': error_value,
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'reconstruction': reconstruction.squeeze(0).cpu(),
            'heatmap': heatmap.squeeze(0).cpu()
        }
```

---

## 5. GradCAM Implementation

### 5.1 Theory

**Gradient-weighted Class Activation Mapping (GradCAM)**

GradCAM produces visual explanations by:
1. Computing gradients of target class score w.r.t. feature maps
2. Global average pooling gradients to get importance weights
3. Weighted combination of feature maps
4. ReLU to keep positive contributions

**Mathematical Formulation**:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

$$L_{GradCAM}^c = ReLU\left(\sum_k \alpha_k^c A^k\right)$$

Where:
- $y^c$ = score for class $c$ (before softmax)
- $A^k$ = activation map of $k$-th channel
- $\alpha_k^c$ = importance weight of channel $k$ for class $c$
- $Z$ = number of pixels in feature map

### 5.2 Implementation

```python
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Highlights image regions most important for classification.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: CNN model
            target_layer: Layer to visualize (typically last conv)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture activations and gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Forward hook: save feature map activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Backward hook: save gradients w.r.t. activations."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate GradCAM heatmap.
        
        Args:
            input_tensor: (1, 3, H, W) input image
            target_class: Class index (None = use predicted class)
        
        Returns:
            cam: (H, W) numpy array, normalized [0, 1]
            predicted_class: int
            confidence: float
        """
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        if target_class is None:
            target_class = predicted_class.item()
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients → channel weights
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, 
                         device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), predicted_class.item(), confidence.item()
```

### 5.3 Overlay Generation

```python
def _create_gradcam_overlay(self, original_image, heatmap, alpha=0.5):
    """
    Create GradCAM overlay on original image.
    
    Args:
        original_image: (H, W, 3) numpy array, uint8 RGB
        heatmap: (h, w) numpy array, float [0, 1]
        alpha: Overlay transparency (0 = only original, 1 = only heatmap)
    
    Returns:
        overlay: (H, W, 3) numpy array, uint8 RGB
    """
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, 
                                  (original_image.shape[1], original_image.shape[0]))
    
    # Convert to colormap (COLORMAP_JET: blue→green→red)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original
    overlay = cv2.addWeighted(original_image, 1 - alpha, 
                              heatmap_colored, alpha, 0)
    
    return overlay
```

---

## 6. Loss Functions & Metrics

### 6.1 CNN Training Loss

**Cross-Entropy Loss with Label Smoothing**:

$$L_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 6.2 CAE Training Loss

**Mean Squared Error (Reconstruction Loss)**:

$$L_{MSE} = \frac{1}{N \cdot C \cdot H \cdot W} \sum_{i,c,h,w} (x_{i,c,h,w} - \hat{x}_{i,c,h,w})^2$$

```python
criterion = nn.MSELoss()
loss = criterion(reconstruction, input_image)
```

### 6.3 Anomaly Detection Metrics

**AUC-ROC (Area Under ROC Curve)**:

Measures ability to distinguish normal from anomaly across all thresholds.

$$AUC = \int_0^1 TPR(FPR^{-1}(t)) \, dt$$

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labels, errors)  # labels: 0=normal, 1=anomaly
```

**F1 Score (at optimal threshold)**:

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

```python
from sklearn.metrics import f1_score
predictions = (errors > threshold).astype(int)
f1 = f1_score(labels, predictions)
```

**Optimal Threshold Selection**:

```python
def find_optimal_threshold(errors, labels):
    """
    Find threshold that maximizes F1 score.
    
    Args:
        errors: Reconstruction errors (normal should be lower)
        labels: Ground truth (0=normal, 1=anomaly)
    
    Returns:
        optimal_threshold: float
        best_f1: float
    """
    best_f1 = 0
    optimal_threshold = np.median(errors)
    
    # Search over threshold range
    for threshold in np.linspace(errors.min(), errors.max(), 100):
        predictions = (errors > threshold).astype(int)
        f1 = f1_score(labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold
    
    return optimal_threshold, best_f1
```

---

## 7. Training Procedures

### 7.1 CNN Training

```python
# Configuration
CONFIG = {
    'epochs': 30,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'early_stopping_patience': 10
}

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',        # Maximize validation accuracy
    factor=0.5,        # Reduce LR by half
    patience=5         # Wait 5 epochs before reducing
)

# Training loop
for epoch in range(CONFIG['epochs']):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    val_accuracy = validate(model, val_loader)
    scheduler.step(val_accuracy)
    
    # Save best model
    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_model.pth')
```

### 7.2 CAE Training

```python
# Configuration
CONFIG = {
    'epochs': 50,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'latent_dim': 128
}

# Training loop
for epoch in range(CONFIG['epochs']):
    model.train()
    for images, _ in train_loader:  # No labels for unsupervised
        images = images.to(device)
        
        reconstruction = model(images)
        loss = criterion(reconstruction, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation: compute AUC on labeled data
    val_auc, threshold = validate_anomaly_detection(model, val_loader)
    scheduler.step(val_auc)
    
    # Save best model
    if val_auc > best_auc:
        torch.save({
            'model_state_dict': model.state_dict(),
            'threshold': threshold,
            'auc': val_auc
        }, 'best_cae.pth')
```

---

## 8. GUI Implementation

**File**: `gui/main.py`

### 8.1 Main Window Structure

```python
class WeldDefectGUI(QMainWindow):
    """
    Main GUI window for weld defect detection.
    
    Components:
    - Tab widget with 5 tabs
    - Control panel with buttons
    - Settings panel
    - Status bar
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIAWELC - Weld Defect Detection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # State variables
        self.image_list = []           # List of image paths
        self.current_image_index = 0   # Current image index
        self.batch_results_cache = {}  # Cached analysis results
        self.current_result = None     # Current analysis result
        
        # Initialize components
        self.config = GUIConfig()
        self.engine = DefectDetectionEngine(self.config)
        
        self._setup_ui()
        self._apply_dark_theme()
```

### 8.2 Tab Structure

```python
def _setup_tabs(self):
    """Create the 5 analysis tabs."""
    self.tabs = QTabWidget()
    
    # Tab 1: Input Image
    self.input_tab = self._create_image_tab(
        "Shows the original input image. "
        "Use Load Image or Load Folder to select images for analysis."
    )
    
    # Tab 2: CNN GradCAM
    self.gradcam_tab = self._create_image_tab(
        "CNN classification with GradCAM visualization. "
        "Red regions indicate areas most important for the prediction."
    )
    
    # Tab 3: AE Reconstruction
    self.ae_recon_tab = self._create_image_tab(
        "Autoencoder reconstruction output. "
        "Compare with input - differences indicate anomalies."
    )
    
    # Tab 4: AE Anomaly Map
    self.ae_anomaly_tab = self._create_image_tab(
        "Pixel-wise reconstruction error heatmap. "
        "Brighter regions have higher error (potential defects)."
    )
    
    # Tab 5: Results
    self.results_tab = self._create_results_tab()
    
    self.tabs.addTab(self.input_tab, "Input Image")
    self.tabs.addTab(self.gradcam_tab, "CNN GradCAM")
    self.tabs.addTab(self.ae_recon_tab, "AE Reconstruction")
    self.tabs.addTab(self.ae_anomaly_tab, "AE Anomaly Map")
    self.tabs.addTab(self.results_tab, "Results")
```

### 8.3 Batch Processing

```python
def load_folder(self):
    """
    Load folder with subfolders CR/, LP/, ND/, PO/.
    Extracts ground truth from folder structure.
    """
    folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
    if not folder:
        return
    
    self.image_list = []
    self.batch_results_cache = {}
    
    # Scan subfolders
    for class_name in ['CR', 'LP', 'ND', 'PO']:
        class_folder = Path(folder) / class_name
        if class_folder.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                for img_path in class_folder.glob(ext):
                    self.image_list.append({
                        'path': str(img_path),
                        'ground_truth': class_name
                    })
    
    self.image_list = sorted(self.image_list, key=lambda x: x['path'])
    self.current_image_index = 0
    self._update_navigation()
    self._load_current_image()


def analyze_all(self):
    """
    Analyze all images in batch with progress dialog.
    Results are cached for quick navigation.
    """
    if not self.image_list:
        return
    
    progress = QProgressDialog("Analyzing images...", "Cancel", 
                               0, len(self.image_list), self)
    progress.setWindowModality(Qt.WindowModal)
    
    for i, img_info in enumerate(self.image_list):
        if progress.wasCanceled():
            break
        
        progress.setValue(i)
        
        # Skip if already cached
        if img_info['path'] in self.batch_results_cache:
            continue
        
        # Analyze and cache
        result = self.engine.predict(img_info['path'], self.config)
        self.batch_results_cache[img_info['path']] = result
    
    progress.setValue(len(self.image_list))
    self._display_current_result()
```

### 8.4 Dark Theme

```python
def _apply_dark_theme(self):
    """Apply dark theme stylesheet."""
    self.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        QPushButton {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
        }
        QPushButton:pressed {
            background-color: #2d2d2d;
        }
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background-color: #252526;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #ffffff;
            padding: 10px 20px;
            border: 1px solid #3c3c3c;
        }
        QTabBar::tab:selected {
            background-color: #094771;
        }
        QLabel {
            color: #ffffff;
        }
        QGroupBox {
            color: #ffffff;
            border: 1px solid #3c3c3c;
            margin-top: 10px;
            padding-top: 10px;
        }
        QCheckBox {
            color: #ffffff;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
        }
    """)
```

---

## 9. Configuration Reference

**File**: `gui/config.py`

```python
class GUIConfig:
    """
    Configuration for GUI application.
    
    Attributes:
        cnn_model_path: Path to trained CNN model
        autoencoder_model_path: Path to trained CAE model
        use_cnn: Enable CNN classification
        use_autoencoder: Enable CAE anomaly detection
        ae_threshold: Anomaly decision threshold
        apply_gaussian: Apply Gaussian blur preprocessing
    """
    
    def __init__(self):
        # Model paths (absolute)
        base_path = Path(__file__).parent.parent
        self.cnn_model_path = str(base_path / 'models' / 'best_model_pytorch.pth')
        self.autoencoder_model_path = str(base_path / 'cae' / 'models' / 'cae_final.pth')
        
        # Analysis settings
        self.use_cnn = True
        self.use_autoencoder = False
        self.ae_threshold = 0.001
        self.apply_gaussian = False
        
        # Display settings
        self.image_size = 224
        self.display_size = 512
```

---

## 10. Troubleshooting & Debugging

### 10.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | GPU memory full | Reduce batch_size or use CPU |
| `Model not found` | Wrong path | Check config.py paths |
| `No images found` | Wrong folder structure | Ensure CR/, LP/, ND/, PO/ subfolders |
| `RuntimeError: grad` | Multiple backward passes | Add `retain_graph=True` |
| `Shape mismatch` | Wrong image size | Check transforms use 224×224 |

### 10.2 Debugging Tools

```python
# Check model loading
def verify_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check data pipeline
def verify_dataloader(loader):
    batch = next(iter(loader))
    if len(batch) == 2:
        images, paths = batch
        print(f"Images: {images.shape}, dtype={images.dtype}")
        print(f"Range: [{images.min():.3f}, {images.max():.3f}]")
    elif len(batch) == 4:
        images, labels, types, paths = batch
        print(f"Images: {images.shape}")
        print(f"Labels: {labels[:5]}")
        print(f"Types: {types[:5]}")

# Check GradCAM
def verify_gradcam(model, image_tensor):
    target_layer = model.features[19]
    gradcam = GradCAM(model, target_layer)
    cam, pred, conf = gradcam.generate_cam(image_tensor)
    print(f"CAM shape: {cam.shape}")
    print(f"CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    print(f"Prediction: {pred}, Confidence: {conf:.3f}")
```

### 10.3 Performance Profiling

```python
import time

def profile_inference(engine, image_path, n_runs=10):
    """Profile inference time."""
    times = []
    
    # Warmup
    _ = engine.predict(image_path)
    
    # Timed runs
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = engine.predict(image_path)
        times.append(time.perf_counter() - start)
    
    print(f"Inference Time (n={n_runs}):")
    print(f"  Mean: {np.mean(times)*1000:.1f} ms")
    print(f"  Std:  {np.std(times)*1000:.1f} ms")
    print(f"  Min:  {np.min(times)*1000:.1f} ms")
    print(f"  Max:  {np.max(times)*1000:.1f} ms")
```

---

## Appendix: File Checksums

For verification of model integrity:

```python
import hashlib

def get_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

# Example usage
# print(get_file_hash('models/best_model_pytorch.pth'))
```

---

*Technical Documentation v1.0 - February 2026*
