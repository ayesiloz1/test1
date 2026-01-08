"""
GradCAM Visualization for Weld Defect Detection
Visualizes model decision-making process using Gradient-weighted Class Activation Mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model settings
    MODEL_PATH = 'models/best_model_pytorch.pth'
    IMG_SIZE = 224
    CLASSES = ['CR', 'LP', 'ND', 'PO']
    NUM_CLASSES = 4
    
    # Dataset
    TEST_DIR = 'dataset/testing'
    
    # Visualization settings
    OUTPUT_DIR = 'gradcam_outputs'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODEL ARCHITECTURES (Must match training script)
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN - Good starting point"""
    
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DeepCNN(nn.Module):
    """Deep CNN - Better accuracy"""
    
    def __init__(self, num_classes=4):
        super(DeepCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(0.4),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================================
# GRADCAM IMPLEMENTATION
# ============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: The layer to compute GradCAM (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate GradCAM heatmap
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index (if None, use predicted class)
            
        Returns:
            cam: GradCAM heatmap
            predicted_class: Predicted class index
            confidence: Prediction confidence
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get prediction
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        # If target class not specified, use predicted class
        if target_class is None:
            target_class = predicted_class
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), predicted_class, confidence
    
    def __call__(self, input_tensor, target_class=None):
        return self.generate_cam(input_tensor, target_class)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """
    Apply heatmap on original image
    
    Args:
        org_img: Original image (numpy array, RGB)
        activation_map: GradCAM heatmap
        colormap: OpenCV colormap
        
    Returns:
        Combined image with heatmap overlay
    """
    # Resize activation map to match image size
    height, width = org_img.shape[:2]
    activation_map = cv2.resize(activation_map, (width, height))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on original image
    overlayed_img = heatmap * 0.4 + org_img * 0.6
    overlayed_img = np.clip(overlayed_img, 0, 255).astype(np.uint8)
    
    return heatmap, overlayed_img

def visualize_gradcam(image_path, model, gradcam, save_path=None):
    """
    Complete GradCAM visualization pipeline
    
    Args:
        image_path: Path to input image
        model: PyTorch model
        gradcam: GradCAM object
        save_path: Path to save visualization
    """
    # Load and preprocess image
    original_img = Image.open(image_path).convert('RGB')
    original_img_np = np.array(original_img)
    
    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(original_img).unsqueeze(0).to(Config.DEVICE)
    
    # Generate GradCAM
    cam, predicted_class, confidence = gradcam(input_tensor)
    
    # Prepare original image for visualization
    img_for_viz = cv2.resize(original_img_np, (Config.IMG_SIZE, Config.IMG_SIZE))
    
    # Apply colormap
    heatmap, overlayed_img = apply_colormap_on_image(img_for_viz, cam)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_for_viz)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlayed_img)
    axes[2].set_title(f'Prediction: {Config.CLASSES[predicted_class]}\n'
                     f'Confidence: {confidence:.2%}', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    
    plt.close()
    
    return predicted_class, confidence

def visualize_all_test_images(model, gradcam):
    """
    Visualize ALL test images and organize by class
    
    Args:
        model: PyTorch model
        gradcam: GradCAM object
    """
    # Create output directory structure
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    for class_name in Config.CLASSES:
        class_dir = os.path.join(Config.OUTPUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Get all test images
    test_dataset = datasets.ImageFolder(Config.TEST_DIR)
    
    print(f"\nProcessing ALL {len(test_dataset.samples)} test images...\n")
    
    # Track statistics
    correct_predictions = 0
    total = len(test_dataset.samples)
    
    # Process each image
    for idx, (image_path, true_label) in enumerate(test_dataset.samples):
        true_class = Config.CLASSES[true_label]
        
        print(f"[{idx+1}/{total}] Class: {true_class} | {os.path.basename(image_path)}", end=" ")
        
        # Generate filename in class-specific folder
        save_path = os.path.join(
            Config.OUTPUT_DIR,
            true_class,
            f"gradcam_{os.path.basename(image_path)}"
        )
        
        # Visualize
        pred_class, confidence = visualize_gradcam(image_path, model, gradcam, save_path)
        pred_class_name = Config.CLASSES[pred_class]
        
        # Track accuracy
        if pred_class == true_label:
            correct_predictions += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"→ {pred_class_name} ({confidence:.2%}) {status}")
    
    # Print summary
    accuracy = correct_predictions / total * 100
    print("\n" + "=" * 70)
    print(f"Processing Complete!")
    print(f"Total Images: {total}")
    print(f"Correct Predictions: {correct_predictions}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Output saved to: {Config.OUTPUT_DIR}")
    print("=" * 70)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path, model_type='auto'):
    """
    Load trained model
    
    Args:
        model_path: Path to saved model
        model_type: 'simple', 'deep', or 'auto' (try to detect)
        
    Returns:
        model: Loaded PyTorch model
        target_layer: Target layer for GradCAM
    """
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    
    # Try to detect model type if auto
    if model_type == 'auto':
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        else:
            # Try to infer from state dict
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            if any('512' in str(k) for k in state_dict.keys()):
                model_type = 'deep'
            else:
                model_type = 'simple'
    
    # Initialize model
    if model_type == 'simple':
        model = SimpleCNN(num_classes=Config.NUM_CLASSES)
        # Target last conv layer in features
        target_layer = model.features[-4]  # Last Conv2d before pooling
    elif model_type == 'deep':
        model = DeepCNN(num_classes=Config.NUM_CLASSES)
        # Target last conv layer in features
        target_layer = model.features[-4]  # Last Conv2d before pooling
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(Config.DEVICE)
    model.eval()
    
    print(f"Model type: {model_type}")
    print(f"Target layer for GradCAM: {target_layer}")
    print(f"Device: {Config.DEVICE}\n")
    
    return model, target_layer

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("GradCAM Visualization for Weld Defect Detection")
    print("=" * 70)
    print()
    
    # Load model
    model, target_layer = load_model(Config.MODEL_PATH, model_type='auto')
    
    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Automatically visualize ALL test images
    print("\nAutomatically processing all test images...\n")
    visualize_all_test_images(model, gradcam)
    
    print("\n" + "=" * 70)
    print("All visualizations saved to:", Config.OUTPUT_DIR)
    print("=" * 70)

if __name__ == "__main__":
    main()
