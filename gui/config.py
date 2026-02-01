"""
GUI Configuration
"""

from pathlib import Path
import os


class GUIConfig:
    """Configuration for GUI application"""
    
    def __init__(self):
        # Get the directory where this script is located
        self.gui_dir = Path(__file__).parent.resolve()
        self.project_root = self.gui_dir.parent
        
        # Model paths (use absolute paths)
        self.cnn_model_path = self.project_root / 'models' / 'best_model_pytorch.pth'
        self.autoencoder_model_path = self.project_root / 'autoencoder_models' / 'best_autoencoder.pth'
        
        # Detection thresholds
        self.ae_threshold = 0.005  # Autoencoder reconstruction error threshold
        self.cnn_threshold = 0.5  # CNN confidence threshold
        
        # Model selection - CNN only by default (SimpleCNN)
        self.use_autoencoder = False  # Set to True when autoencoder is available
        self.use_cnn = True
        
        # Processing options
        self.apply_gaussian = False
        self.save_results = False
        
        # Output directories
        self.output_dir = Path('gui_outputs')
        self.output_dir.mkdir(exist_ok=True)
        
        # Image settings
        self.image_size = (224, 224)
        
        # Theme
        self.theme = 'dark'  # 'dark' or 'light'
        
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'cnn_model_path': str(self.cnn_model_path),
            'autoencoder_model_path': str(self.autoencoder_model_path),
            'ae_threshold': self.ae_threshold,
            'cnn_threshold': self.cnn_threshold,
            'use_autoencoder': self.use_autoencoder,
            'use_cnn': self.use_cnn,
            'apply_gaussian': self.apply_gaussian,
        }
