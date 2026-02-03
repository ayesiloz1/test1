"""
Image Processing Utilities
"""

import numpy as np
import cv2
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


def display_tensor_image(tensor_or_array, label):
    """Convert tensor or numpy array to QPixmap and display"""
    try:
        # Handle both torch tensors and numpy arrays
        if hasattr(tensor_or_array, 'cpu'):
            # It's a torch tensor
            tensor_or_array = tensor_or_array.cpu()
            if hasattr(tensor_or_array, 'numpy'):
                tensor_or_array = tensor_or_array.numpy()
        
        # Now we have a numpy array
        img_array = tensor_or_array
        
        # Handle different shapes
        if len(img_array.shape) == 3:
            # (C, H, W) -> (H, W, C)
            if img_array.shape[0] == 3 or img_array.shape[0] == 1:
                img_array = np.transpose(img_array, (1, 2, 0))
        
        # Normalize to [0, 255]
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        
        # Ensure contiguous array for QImage
        img_array = np.ascontiguousarray(img_array)
        
        # Convert to QImage
        if len(img_array.shape) == 2:  # Grayscale
            height, width = img_array.shape
            # Apply colormap for better visualization
            img_colored = cv2.applyColorMap(img_array, cv2.COLORMAP_JET)
            img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            qimage = QImage(img_colored.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        else:  # RGB
            if img_array.shape[2] == 1:
                img_array = img_array.squeeze(2)
                # Apply colormap
                img_colored = cv2.applyColorMap(img_array, cv2.COLORMAP_JET)
                img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)
                height, width, _ = img_colored.shape
                bytes_per_line = 3 * width
                qimage = QImage(img_colored.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                height, width, channels = img_array.shape
                bytes_per_line = channels * width
                qimage = QImage(img_array.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
    except Exception as e:
        label.setText(f"Error displaying image: {str(e)}")
        print(f"Error in display_tensor_image: {e}")


def display_numpy_image(image_array, label):
    """Display numpy array as image in label"""
    try:
        if len(image_array.shape) == 2:
            # Grayscale
            height, width = image_array.shape
            bytes_per_line = width
            qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # RGB
            height, width, channels = image_array.shape
            bytes_per_line = channels * width
            if channels == 3:
                qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                return
        
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
    except Exception as e:
        label.setText(f"Error displaying image: {str(e)}")
        print(f"Error in display_numpy_image: {e}")
