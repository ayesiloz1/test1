"""
Visualization Utilities for Defect Detection
"""

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms


def create_defect_detection_view(image_path, result):
    """
    Create combined visualization with GradCAM + AE bounding boxes
    
    Args:
        image_path: Path to the original image
        result: Detection result dictionary with 'cnn' and 'autoencoder' keys
        
    Returns:
        tuple: (combined_image_array, summary_text)
    """
    try:
        # Load original image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Start with original image as base
        combined = img_array.astype(np.float32) / 255.0
        
        # Apply GradCAM overlay if available (CNN - Supervised)
        if result['cnn'] and result['cnn'].get('gradcam_heatmap') is not None:
            gradcam = result['cnn']['gradcam_heatmap']
            gradcam_color = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
            gradcam_color = cv2.cvtColor(gradcam_color, cv2.COLOR_BGR2RGB) / 255.0
            combined = combined * 0.6 + gradcam_color * 0.4
        
        # Calculate bounding boxes from AE reconstruction difference
        bounding_boxes = []
        if result['autoencoder'] and result['autoencoder'].get('reconstruction') is not None:
            bounding_boxes = _calculate_anomaly_bounding_boxes(img, result['autoencoder']['reconstruction'])
        
        # Convert to uint8
        combined = np.clip(combined * 255, 0, 255).astype(np.uint8)
        
        # Resize for display
        combined = cv2.resize(combined, (600, 600))
        
        # Draw bounding boxes
        if bounding_boxes:
            scale = 600 / 224
            for i, (x, y, w, h) in enumerate(bounding_boxes):
                x_scaled = int(x * scale)
                y_scaled = int(y * scale)
                w_scaled = int(w * scale)
                h_scaled = int(h * scale)
                
                cv2.rectangle(combined, (x_scaled, y_scaled),
                            (x_scaled + w_scaled, y_scaled + h_scaled),
                            (0, 255, 0), 2)
                
                if i == 0 and len(bounding_boxes) > 0:
                    cv2.putText(combined, "Anomaly Region",
                              (x_scaled, max(y_scaled - 10, 20)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Generate summary text
        summary = _generate_summary(result)
        
        return combined, summary
        
    except Exception as e:
        print(f"Error in create_defect_detection_view: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"


def _calculate_anomaly_bounding_boxes(img, reconstruction):
    """Calculate bounding boxes from reconstruction difference"""
    try:
        # Get original image tensor
        original_tensor = transforms.ToTensor()(img).unsqueeze(0).to('cpu')
        
        # Apply same normalization as model input
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        original_normalized = normalize(original_tensor.squeeze(0)).unsqueeze(0)
        
        # Get reconstruction
        reconstruction = reconstruction.unsqueeze(0).cpu()
        
        # Calculate absolute difference
        diff = torch.abs(original_normalized - reconstruction)
        diff = torch.mean(diff, dim=1).squeeze().numpy()
        
        # Resize to 224x224
        diff = cv2.resize(diff, (224, 224))
        
        # Normalize difference to 0-1
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        
        # Use percentile-based threshold
        threshold = np.percentile(diff, 95)  # Top 5% differences
        binary_mask = (diff > threshold).astype(np.uint8)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove edge artifacts
        binary_mask[0:10, :] = 0
        binary_mask[-10:, :] = 0
        binary_mask[:, 0:10] = 0
        binary_mask[:, -10:] = 0
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter noise
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append((x, y, w, h))
        
        return bounding_boxes
        
    except Exception as e:
        print(f"Error calculating bounding boxes: {e}")
        return []


def _generate_summary(result):
    """Generate text summary of detection results"""
    lines = []
    
    # CNN results
    if result.get('cnn'):
        cnn = result['cnn']
        lines.append(f"CNN Prediction: {cnn['prediction']} ({cnn['confidence']:.1%})")
    
    # Autoencoder results
    if result.get('autoencoder'):
        ae = result['autoencoder']
        lines.append(f"\nAutoencoder: {ae['status']}")
        lines.append(f"Error: {ae['error']:.6f} (Threshold: {ae['threshold']:.6f})")
    
    # Processing time
    if result.get('processing_time'):
        lines.append(f"\nProcessing Time: {result['processing_time']:.3f}s")
    
    return "\n".join(lines)
