"""
Image Viewer Widget - Center panel with tabs
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTabWidget, QProgressBar
)
from PyQt5.QtCore import Qt


class ImageViewer(QWidget):
    """Center panel with image display tabs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Image size
        IMG_SIZE = 580
        
        # Create tabs
        self._create_input_tab(IMG_SIZE)
        self._create_gradcam_tab(IMG_SIZE)
        self._create_reconstruction_tab(IMG_SIZE)
        self._create_anomaly_map_tab(IMG_SIZE)
        self._create_results_tab(IMG_SIZE)
        
        layout.addWidget(self.tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
    
    def _create_input_tab(self, img_size):
        """Create input image tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(5)
        
        self.lbl_ground_truth = self._create_ground_truth_label()
        layout.addWidget(self.lbl_ground_truth)
        
        self.lbl_original_image = QLabel()
        self.lbl_original_image.setAlignment(Qt.AlignCenter)
        self.lbl_original_image.setFixedSize(img_size, img_size - 70)
        self.lbl_original_image.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.lbl_original_image.setText("No Image Loaded")
        layout.addWidget(self.lbl_original_image, alignment=Qt.AlignCenter)
        
        info = QLabel("Original input image - Load an image or folder to begin analysis")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #aaa; font-size: 20px; padding: 5px;")
        layout.addWidget(info)
        
        self.tab_widget.addTab(tab, "Input")
    
    def _create_gradcam_tab(self, img_size):
        """Create GradCAM tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(5)
        
        self.lbl_ground_truth_gradcam = self._create_ground_truth_label()
        layout.addWidget(self.lbl_ground_truth_gradcam)
        
        self.lbl_gradcam_image = QLabel()
        self.lbl_gradcam_image.setAlignment(Qt.AlignCenter)
        self.lbl_gradcam_image.setFixedSize(img_size, img_size - 70)
        self.lbl_gradcam_image.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.lbl_gradcam_image.setText("CNN GradCAM visualization\\n(Supervised mode)")
        layout.addWidget(self.lbl_gradcam_image, alignment=Qt.AlignCenter)
        
        info = QLabel("GradCAM highlights regions the CNN uses to make predictions - Red/yellow areas indicate high importance")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #aaa; font-size: 20px; padding: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        self.tab_widget.addTab(tab, "CNN GradCAM")
    
    def _create_reconstruction_tab(self, img_size):
        """Create reconstruction tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(5)
        
        self.lbl_ground_truth_recon = self._create_ground_truth_label()
        layout.addWidget(self.lbl_ground_truth_recon)
        
        self.lbl_reconstructed_image = QLabel()
        self.lbl_reconstructed_image.setAlignment(Qt.AlignCenter)
        self.lbl_reconstructed_image.setFixedSize(img_size, img_size - 70)
        self.lbl_reconstructed_image.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.lbl_reconstructed_image.setText("Autoencoder reconstruction\\n(Unsupervised mode)")
        layout.addWidget(self.lbl_reconstructed_image, alignment=Qt.AlignCenter)
        
        info = QLabel("Autoencoder attempts to reconstruct the input - Poor reconstruction indicates anomaly/defect")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #aaa; font-size: 20px; padding: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        self.tab_widget.addTab(tab, "AE Reconstruction")
    
    def _create_anomaly_map_tab(self, img_size):
        """Create anomaly heatmap tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(5)
        
        self.lbl_ground_truth_heatmap = self._create_ground_truth_label()
        layout.addWidget(self.lbl_ground_truth_heatmap)
        
        self.lbl_heatmap_image = QLabel()
        self.lbl_heatmap_image.setAlignment(Qt.AlignCenter)
        self.lbl_heatmap_image.setFixedSize(img_size, img_size - 70)
        self.lbl_heatmap_image.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.lbl_heatmap_image.setText("Anomaly heatmap\\n(Unsupervised mode)")
        layout.addWidget(self.lbl_heatmap_image, alignment=Qt.AlignCenter)
        
        info = QLabel("Pixel-wise reconstruction error - Bright areas show where the autoencoder failed to reconstruct (potential defects)")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #aaa; font-size: 20px; padding: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        self.tab_widget.addTab(tab, "AE Anomaly Map")
    
    def _create_results_tab(self, img_size):
        """Create combined results tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        self.lbl_ground_truth_results = self._create_ground_truth_label()
        layout.addWidget(self.lbl_ground_truth_results)
        
        self.lbl_defect_detection = QLabel()
        self.lbl_defect_detection.setAlignment(Qt.AlignCenter)
        self.lbl_defect_detection.setFixedSize(img_size, img_size - 70)
        self.lbl_defect_detection.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.lbl_defect_detection.setText("Combined detection results")
        layout.addWidget(self.lbl_defect_detection, alignment=Qt.AlignCenter)
        
        info = QLabel("CNN GradCAM overlay (red/yellow = attention) + AE bounding box (green = anomaly region)")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #aaa; font-size: 20px; padding: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        self.tab_widget.addTab(tab, "Results")
    
    def _create_ground_truth_label(self):
        """Create ground truth label with default styling"""
        label = QLabel("Ground Truth: Unknown")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 8px;
            font-size: 13px;
            font-weight: bold;
        """)
        label.setFixedHeight(35)
        return label
    
    def update_ground_truth(self, class_name):
        """Update all ground truth labels"""
        class_full_names = {
            'CR': 'Crack',
            'LP': 'Lack of Penetration',
            'ND': 'No Defect',
            'PO': 'Porosity'
        }
        
        labels = [
            self.lbl_ground_truth,
            self.lbl_ground_truth_gradcam,
            self.lbl_ground_truth_recon,
            self.lbl_ground_truth_heatmap,
            self.lbl_ground_truth_results
        ]
        
        if class_name and class_name in class_full_names:
            full_name = class_full_names[class_name]
            color = "#44ff44" if class_name == 'ND' else "#ff9944"
            text = f"Ground Truth: {class_name} ({full_name})"
            style = f"""
                background-color: #2b2b2b;
                border: 2px solid {color};
                border-radius: 3px;
                padding: 8px;
                font-size: 13px;
                font-weight: bold;
                color: {color};
            """
        else:
            text = "Ground Truth: Unknown"
            style = """
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 8px;
                font-size: 13px;
                font-weight: bold;
                color: #888;
            """
        
        for label in labels:
            label.setText(text)
            label.setStyleSheet(style)
    
    def clear_results(self):
        """Clear all result visualizations"""
        self.lbl_gradcam_image.clear()
        self.lbl_gradcam_image.setText("Click 'Analyze Image' to process")
        self.lbl_reconstructed_image.clear()
        self.lbl_reconstructed_image.setText("Not analyzed")
        self.lbl_heatmap_image.clear()
        self.lbl_heatmap_image.setText("Not analyzed")
        self.lbl_defect_detection.clear()
        self.lbl_defect_detection.setText("Not analyzed")
