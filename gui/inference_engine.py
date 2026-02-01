"""
Inference Engine - Combines CNN and Autoencoder for Defect Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from train_pytorch import SimpleCNN

# Try to import autoencoder, but make it optional
try:
    from autoencoder_src.model import ConvolutionalAutoencoder
    AUTOENCODER_AVAILABLE = True
except ImportError:
    AUTOENCODER_AVAILABLE = False
    print("Note: Autoencoder module not found. Using CNN-only mode.")


class GradCAM:
    """Gradient-weighted Class Activation Mapping for visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate GradCAM heatmap"""
        self.model.eval()
        
        # Enable gradients for this forward pass
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        if target_class is None:
            target_class = predicted_class
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = torch.mean(gradients, dim=(1, 2))
        
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), predicted_class, confidence


class DefectDetectionEngine:
    """Combined detection engine using both CNN and Autoencoder"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.cnn_model = None
        self.autoencoder_model = None
        self.gradcam = None
        
        self._load_cnn()
        if AUTOENCODER_AVAILABLE:
            self._load_autoencoder()
        
        # Class names
        self.class_names = ['CR', 'LP', 'ND', 'PO']
        
    def _load_cnn(self):
        """Load CNN classifier (SimpleCNN)"""
        try:
            cnn_path = self.config.cnn_model_path
            if not Path(cnn_path).exists():
                print(f"Warning: CNN model not found at {cnn_path}")
                return
                
            checkpoint = torch.load(cnn_path, map_location=self.device, weights_only=False)
            
            self.cnn_model = SimpleCNN(num_classes=4)
            
            # Handle both formats: direct state_dict or checkpoint dictionary
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Direct state_dict format
                self.cnn_model.load_state_dict(checkpoint)
            
            self.cnn_model.to(self.device)
            self.cnn_model.eval()
            
            # Setup GradCAM with the last conv layer (features[-4] is the last Conv2d before final pooling)
            # For SimpleCNN, the last conv block is at index 19 (nn.Conv2d(128, 128))
            target_layer = self.cnn_model.features[19]  # Last conv layer in Block 3
            self.gradcam = GradCAM(self.cnn_model, target_layer)
            
            print(f"CNN model loaded from {cnn_path}")
            print(f"GradCAM initialized with target layer: {target_layer}")
        except Exception as e:
            print(f"Error loading CNN: {e}")
            self.cnn_model = None
            
    def _load_autoencoder(self):
        """Load autoencoder"""
        try:
            ae_path = self.config.autoencoder_model_path
            if not Path(ae_path).exists():
                print(f"Warning: Autoencoder model not found at {ae_path}")
                return
                
            checkpoint = torch.load(ae_path, map_location=self.device, weights_only=False)
            
            self.autoencoder_model = ConvolutionalAutoencoder(
                input_channels=checkpoint['config']['input_channels'],
                latent_dim=checkpoint['config']['latent_dim'],
                encoder_channels=checkpoint['config']['encoder_channels']
            )
            self.autoencoder_model.load_state_dict(checkpoint['model_state_dict'])
            self.autoencoder_model.to(self.device)
            self.autoencoder_model.eval()
            
            # Get threshold from checkpoint
            self.ae_threshold = checkpoint.get('threshold', self.config.ae_threshold)
            
            print(f"✓ Autoencoder model loaded from {ae_path}")
            print(f"  Default threshold: {self.ae_threshold:.6f}")
        except Exception as e:
            print(f"Error loading Autoencoder: {e}")
            self.autoencoder_model = None
            
    def get_transforms(self):
        """Get image preprocessing transforms"""
        transform_list = [transforms.Resize((224, 224))]
        
        if self.config.apply_gaussian:
            transform_list.append(transforms.GaussianBlur(kernel_size=5, sigma=1.0))
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
        
    def predict(self, image_path, config=None):
        """
        Run prediction on image using both models
        
        Returns:
            dict: {
                'autoencoder': {...},
                'cnn': {...},
                'processing_time': float
            }
        """
        if config:
            self.config = config
            
        start_time = time.time()
        
        result = {
            'autoencoder': None,
            'cnn': None,
            'processing_time': 0
        }
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((224, 224))
        original_image_np = np.array(image_resized)  # For GradCAM overlay
        
        transform = self.get_transforms()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Autoencoder prediction
        if self.config.use_autoencoder and self.autoencoder_model:
            result['autoencoder'] = self._predict_autoencoder(image_tensor)
            
        # CNN prediction (run if enabled, regardless of autoencoder result)
        if self.config.use_cnn and self.cnn_model:
            result['cnn'] = self._predict_cnn(image_tensor, original_image_np)
                
        result['processing_time'] = time.time() - start_time
        
        return result
        
    def _predict_autoencoder(self, image_tensor):
        """Autoencoder anomaly detection"""
        with torch.no_grad():
            error, reconstruction = self.autoencoder_model.get_reconstruction_error(
                image_tensor, reduction='none'
            )
            
            error_value = error.item()
            threshold = self.config.ae_threshold or self.ae_threshold
            is_anomaly = error_value > threshold
            
            # Generate heatmap
            diff = torch.abs(image_tensor - reconstruction)
            diff_normalized = diff / (diff.max() + 1e-8)
            heatmap = diff_normalized.mean(dim=1, keepdim=True)  # Average across channels
            
            return {
                'error': error_value,
                'threshold': threshold,
                'is_anomaly': is_anomaly,
                'reconstruction': reconstruction.squeeze(0).cpu(),
                'heatmap': heatmap.squeeze(0).cpu()
            }
            
    def _predict_cnn(self, image_tensor, original_image=None):
        """CNN classification with GradCAM visualization"""
        # First get prediction with no_grad for efficiency
        with torch.no_grad():
            outputs = self.cnn_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            pred_class = self.class_names[predicted.item()]
            conf_value = confidence.item()
            
            # Get all class probabilities
            class_probs = {}
            for i, class_name in enumerate(self.class_names):
                class_probs[class_name] = probabilities[0][i].item()
        
        # Generate GradCAM heatmap (requires gradients)
        gradcam_heatmap = None
        gradcam_overlay = None
        if self.gradcam is not None and original_image is not None:
            try:
                # Need a fresh tensor with gradients enabled
                image_tensor_grad = image_tensor.clone().detach().requires_grad_(True)
                cam, _, _ = self.gradcam.generate_cam(image_tensor_grad, predicted.item())
                
                # Resize CAM to original image size
                cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
                
                # Create heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Create overlay
                overlay = np.float32(heatmap) / 255 + np.float32(original_image) / 255
                overlay = overlay / overlay.max()
                overlay = np.uint8(255 * overlay)
                
                gradcam_heatmap = cam_resized
                gradcam_overlay = overlay
            except Exception as e:
                print(f"GradCAM generation error: {e}")
                
        return {
            'prediction': pred_class,
            'confidence': conf_value,
            'probabilities': class_probs,
            'meets_threshold': conf_value >= self.config.cnn_threshold,
            'gradcam_heatmap': gradcam_heatmap,
            'gradcam_overlay': gradcam_overlay
        }
