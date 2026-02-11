"""
Professional Weld Defect Detection System - Main GUI (Refactored)
Combines CNN Classifier and Autoencoder for comprehensive defect detection
"""

import sys
import os
import json
from pathlib import Path
import time
import torch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QFileDialog, QMessageBox, QStatusBar, QMenuBar, QAction, QSplitter
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from widgets import ControlPanel, ImageViewer, ResultsPanel
from utils import display_tensor_image, display_numpy_image, create_defect_detection_view
from inference_engine import DefectDetectionEngine
from config import GUIConfig
from security_utils import validate_file_upload, validate_path_safety, sanitize_filename


class WeldDefectGUI(QMainWindow):
    """Main GUI window"""
    
    def __init__(self):
        super().__init__()
        self.config = GUIConfig()
        self.engine = None
        self.current_image_path = None
        self.current_result = None
        
        # Batch processing
        self.image_list = []
        self.filtered_image_list = []
        self.current_image_index = 0
        self.batch_results_cache = {}
        self.current_class_filter = "All"
        self.processing_delay = 0
        self.is_paused = False
        self.batch_results = []
        
        self.init_ui()
        self.load_models()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Weld Defect Detection System v1.0")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(self.get_stylesheet())
        
        # Create menu bar
        self.create_menu_bar()
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create panels
        self.control_panel = ControlPanel()
        self.image_viewer = ImageViewer()
        self.results_panel = ResultsPanel()
        
        # Connect signals
        self._connect_signals()
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.image_viewer)
        splitter.addWidget(self.results_panel)
        splitter.setSizes([300, 700, 400])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _connect_signals(self):
        """Connect all signals from widgets"""
        # Control panel signals
        self.control_panel.load_image_clicked.connect(self.load_image)
        self.control_panel.load_folder_clicked.connect(self.load_folder)
        self.control_panel.prev_clicked.connect(self.prev_image)
        self.control_panel.next_clicked.connect(self.next_image)
        self.control_panel.class_filter_changed.connect(self.filter_images_by_class)
        self.control_panel.review_mode_changed.connect(self.filter_by_review_mode)
        self.control_panel.detection_mode_changed.connect(self.on_detection_mode_changed)
        self.control_panel.analyze_clicked.connect(self.analyze_image)
        self.control_panel.analyze_all_clicked.connect(self.analyze_all_images)
        self.control_panel.pause_clicked.connect(self.toggle_pause)
        self.control_panel.speed_changed.connect(self.on_speed_changed)
        self.control_panel.export_clicked.connect(self.export_results)
        self.control_panel.clear_clicked.connect(self.clear_results)
        self.control_panel.clear_all_clicked.connect(self.clear_all_results)
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        open_action = QAction('Open Image', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        open_folder_action = QAction('Open Folder', self)
        open_folder_action.setShortcut('Ctrl+Shift+O')
        open_folder_action.triggered.connect(self.load_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def get_stylesheet(self):
        """Get application stylesheet"""
        return """
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
            }
            QGroupBox {
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0e639c;
                border: none;
                padding: 8px;
                border-radius: 3px;
                color: white;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5689;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #555;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0e639c;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 2px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
            }
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 2px solid #555;
            }
            QTabBar::tab {
                background-color: #2b2b2b;
                padding: 8px 15px;
                border: 1px solid #555;
            }
            QTabBar::tab:selected {
                background-color: #0e639c;
            }
            QStatusBar {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QMenuBar {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #0e639c;
            }
            QMenu {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
            }
            QMenu::item:selected {
                background-color: #0e639c;
            }
        """
    
    def load_models(self):
        """Load detection models"""
        try:
            self.results_panel.log_message("Loading models...")
            self.engine = DefectDetectionEngine(self.config)
            self.results_panel.log_message("Models loaded successfully")
            self.status_bar.showMessage("Models loaded - Ready")
        except Exception as e:
            self.results_panel.log_message(f"✗ Error loading models: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load models:\n{str(e)}")
    
    def load_image(self):
        """Load single image with validation"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        
        if file_path:
            # Validate file upload
            is_valid, error_msg = validate_file_upload(
                file_path,
                allowed_extensions=['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'],
                max_size_mb=10,
                check_content=True
            )
            
            if not is_valid:
                QMessageBox.critical(
                    self, "Invalid File", 
                    f"File validation failed: {error_msg}"
                )
                self.results_panel.log_message(f"✗ Validation failed: {error_msg}")
                return
            
            # Additional path safety check
            try:
                file_path_obj = Path(file_path).resolve()
                if not file_path_obj.exists():
                    raise FileNotFoundError("File does not exist")
            except Exception as e:
                QMessageBox.critical(
                    self, "Invalid Path",
                    f"Path validation failed: {str(e)}"
                )
                return
            
            self.current_image_path = file_path
            self.display_image(file_path)
            self.control_panel.btn_analyze.setEnabled(True)
            self.results_panel.log_message(f"✓ Loaded: {Path(file_path).name}")
            self.status_bar.showMessage(f"Image loaded: {Path(file_path).name}")
    
    def load_folder(self):
        """Load folder for batch processing with validation"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder (with CR, LP, ND, PO subfolders)"
        )
        
        if folder_path:
            # Validate folder path
            try:
                folder = Path(folder_path).resolve()
                if not folder.exists() or not folder.is_dir():
                    raise ValueError("Invalid folder path")
            except Exception as e:
                QMessageBox.critical(
                    self, "Invalid Folder",
                    f"Folder validation failed: {str(e)}"
                )
                return
            
            self.image_list = []
            self.batch_results_cache = {}
            
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
            subfolders = ['CR', 'LP', 'ND', 'PO']
            has_subfolders = any((folder / sf).exists() for sf in subfolders)
            
            if has_subfolders:
                for subfolder in subfolders:
                    subfolder_path = folder / subfolder
                    if subfolder_path.exists():
                        for ext in extensions:
                            self.image_list.extend(subfolder_path.glob(ext))
                self.results_panel.log_message(f"Loaded from subfolders: {', '.join(subfolders)}")
            else:
                for ext in extensions:
                    self.image_list.extend(folder.glob(ext))
            
            self.image_list = sorted([str(p) for p in self.image_list])
            
            if self.image_list:
                self.current_class_filter = "All"
                self.control_panel.combo_class_filter.setCurrentText("All")
                self.filtered_image_list = self.image_list.copy()
                
                self.current_image_index = 0
                self.load_current_image()
                self.update_navigation_buttons()
                self.control_panel.btn_analyze_all.setEnabled(True)
                self.control_panel.combo_class_filter.setEnabled(True)
                self.results_panel.log_message(f"Loaded {len(self.image_list)} images from folder")
                self.status_bar.showMessage(f"Loaded {len(self.image_list)} images")
            else:
                QMessageBox.warning(self, "Warning", "No images found in the selected folder.")
                self.results_panel.log_message("No images found in folder")
    
    def load_current_image(self):
        """Load and display the current image from the filtered list"""
        if self.filtered_image_list and 0 <= self.current_image_index < len(self.filtered_image_list):
            self.current_image_path = self.filtered_image_list[self.current_image_index]
            self.display_image(self.current_image_path)
            self.control_panel.btn_analyze.setEnabled(True)
            
            img_path = Path(self.current_image_path)
            parent_name = img_path.parent.name
            if parent_name in ['CR', 'LP', 'ND', 'PO']:
                display_name = f"[{parent_name}] {img_path.name}"
                self.image_viewer.update_ground_truth(parent_name)
            else:
                display_name = img_path.name
                self.image_viewer.update_ground_truth(None)
            self.status_bar.showMessage(f"Image: {display_name}")
    
    def display_image(self, image_path):
        """Display image in GUI"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            self.image_viewer.lbl_original_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_viewer.lbl_original_image.setPixmap(scaled_pixmap)
        
        # Update ground truth
        img_path = Path(image_path)
        parent_name = img_path.parent.name
        if parent_name in ['CR', 'LP', 'ND', 'PO']:
            self.image_viewer.update_ground_truth(parent_name)
        else:
            self.image_viewer.update_ground_truth(None)
    
    def update_navigation_buttons(self):
        """Update navigation buttons and counter"""
        if self.filtered_image_list:
            self.control_panel.btn_prev.setEnabled(self.current_image_index > 0)
            self.control_panel.btn_next.setEnabled(
                self.current_image_index < len(self.filtered_image_list) - 1
            )
            if self.current_class_filter == "All":
                text = f"{self.current_image_index + 1} / {len(self.filtered_image_list)}"
            else:
                text = f"{self.current_image_index + 1} / {len(self.filtered_image_list)} (Class: {self.current_class_filter})"
            self.control_panel.lbl_image_counter.setText(text)
        else:
            self.control_panel.btn_prev.setEnabled(False)
            self.control_panel.btn_next.setEnabled(False)
            self.control_panel.lbl_image_counter.setText("No images loaded")
    
    def prev_image(self):
        """Go to previous image"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            self.update_navigation_buttons()
            self.display_cached_results()
    
    def next_image(self):
        """Go to next image"""
        if self.current_image_index < len(self.filtered_image_list) - 1:
            self.current_image_index += 1
            self.load_current_image()
            self.update_navigation_buttons()
            self.display_cached_results()
    
    def filter_images_by_class(self, class_name):
        """Filter images by selected class"""
        if not self.image_list:
            return
        
        self.current_class_filter = class_name
        
        if class_name == "All":
            self.filtered_image_list = self.image_list.copy()
        else:
            self.filtered_image_list = [
                img for img in self.image_list
                if Path(img).parent.name == class_name
            ]
        
        if self.filtered_image_list:
            self.current_image_index = 0
            self.load_current_image()
            self.update_navigation_buttons()
            self.display_cached_results()
            self.results_panel.log_message(
                f"Filtered to {len(self.filtered_image_list)} images (Class: {class_name})"
            )
        else:
            self.current_image_index = 0
            self.update_navigation_buttons()
            QMessageBox.information(self, "Filter", f"No images found for class: {class_name}")
    
    def filter_by_review_mode(self, mode_text):
        """Filter images based on review mode (misclassified, low confidence, etc.)"""
        if not self.image_list:
            return
        
        # If "All Images" is selected, just apply class filter
        if mode_text == "All Images":
            current_class = self.control_panel.combo_class_filter.currentText()
            self.filter_images_by_class(current_class)
            return
        
        # Check if batch analysis has been completed
        if not self.batch_results_cache:
            QMessageBox.warning(
                self, 
                "Review Mode",
                "Please run 'Analyze All Images' first to enable review mode."
            )
            self.control_panel.combo_review_mode.setCurrentIndex(0)  # Reset to "All Images"
            return
        
        # Start with all images or current class filter
        current_class = self.control_panel.combo_class_filter.currentText()
        if current_class == "All":
            base_list = self.image_list.copy()
        else:
            base_list = [
                img for img in self.image_list
                if Path(img).parent.name == current_class
            ]
        
        # Apply review mode filter
        filtered_list = []
        
        for img_path in base_list:
            if img_path not in self.batch_results_cache:
                continue
            
            result = self.batch_results_cache[img_path]
            
            # Get ground truth from folder name
            ground_truth = Path(img_path).parent.name
            
            # Skip if not in a class folder
            if ground_truth not in ['CR', 'LP', 'ND', 'PO']:
                continue
            
            # Get prediction and confidence from result structure
            predicted_class = 'Unknown'
            confidence = 0.0
            is_anomaly = False
            
            if result.get('cnn'):
                predicted_class = result['cnn'].get('prediction', 'Unknown')
                confidence = result['cnn'].get('confidence', 0.0)
            
            if result.get('autoencoder'):
                is_anomaly = result['autoencoder'].get('is_anomaly', False)
            
            # For hybrid mode, use hybrid prediction
            if self.config.use_cnn and self.config.use_autoencoder:
                from utils import get_hybrid_prediction
                predicted_class = get_hybrid_prediction(predicted_class, confidence, is_anomaly)
            
            # Apply filter based on mode
            is_misclassified = (predicted_class != ground_truth)
            is_low_confidence = (confidence < 0.70)
            
            if mode_text == "Misclassified Only":
                if is_misclassified:
                    filtered_list.append(img_path)
            elif mode_text == "Low Confidence (<70%)":
                if is_low_confidence:
                    filtered_list.append(img_path)
            elif mode_text == "Issues (Misclassified or Low Conf.)":
                if is_misclassified or is_low_confidence:
                    filtered_list.append(img_path)
        
        # Update filtered list
        self.filtered_image_list = filtered_list
        
        if self.filtered_image_list:
            self.current_image_index = 0
            self.load_current_image()
            self.update_navigation_buttons()
            self.display_cached_results()
            self.results_panel.log_message(
                f"Review Mode: {mode_text} - Found {len(self.filtered_image_list)} images"
            )
        else:
            self.current_image_index = 0
            self.update_navigation_buttons()
            self.results_panel.log_message(
                f"Review Mode: {mode_text} - No images match the criteria"
            )
    
    def display_cached_results(self):
        """Display cached results for current image if available"""
        if self.current_image_path and self.current_image_path in self.batch_results_cache:
            result = self.batch_results_cache[self.current_image_path]
            self.display_results(result)
            self.results_panel.log_message(
                f"Showing cached results for {Path(self.current_image_path).name}"
            )
        else:
            self.image_viewer.clear_results()
    
    def on_detection_mode_changed(self, index):
        """Handle detection mode change"""
        mode_descriptions = [
            "CNN classification with GradCAM visualization.\nClassifies defects into: CR, LP, ND, PO",
            "Autoencoder anomaly detection.\nDetects anomalies by reconstruction error",
            "Combined CNN + Autoencoder.\nAutoencoder screens for anomalies, CNN classifies defects"
        ]
        self.control_panel.lbl_mode_desc.setText(mode_descriptions[index])
        
        if index == 0:  # Supervised (CNN)
            self.config.use_cnn = True
            self.config.use_autoencoder = False
        elif index == 1:  # Unsupervised (Autoencoder)
            self.config.use_cnn = False
            self.config.use_autoencoder = True
        else:  # Hybrid
            self.config.use_cnn = True
            self.config.use_autoencoder = True
        
        self.results_panel.log_message(
            f"Detection mode: {self.control_panel.combo_detection_mode.currentText()}"
        )
    
    def on_speed_changed(self, value):
        """Handle processing speed slider change"""
        self.processing_delay = value / 10.0
        self.control_panel.lbl_speed_value.setText(f"{self.processing_delay:.1f}s")
    
    def toggle_pause(self):
        """Toggle pause/resume for batch processing"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.control_panel.btn_pause.setText("Resume")
            self.control_panel.btn_pause.setStyleSheet(
                "font-size: 12px; background-color: #00aa00;"
            )
            self.results_panel.log_message("Batch processing paused")
        else:
            self.control_panel.btn_pause.setText("Pause")
            self.control_panel.btn_pause.setStyleSheet(
                "font-size: 12px; background-color: #cc7700;"
            )
            self.results_panel.log_message("Batch processing resumed")
    
    def analyze_image(self):
        """Analyze current image"""
        if not self.current_image_path:
            return
        
        if not self.engine:
            QMessageBox.warning(self, "Warning", "Model not loaded!")
            return
        
        try:
            self.results_panel.log_message(f"Analyzing: {Path(self.current_image_path).name}")
            result = self.engine.predict(self.current_image_path, self.config)
            self.current_result = result
            self.display_results(result)
            self.results_panel.log_message("Analysis complete")
        except Exception as e:
            self.results_panel.log_message(f"✗ Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Analysis failed:\n{str(e)}")
    
    def display_results(self, result):
        """Display detection results"""
        # Update autoencoder results
        if result.get('autoencoder'):
            ae_data = result['autoencoder']
            self.results_panel.update_ae_results(ae_data)
            
            # Display reconstruction
            if ae_data.get('reconstruction') is not None:
                display_tensor_image(
                    ae_data['reconstruction'],
                    self.image_viewer.lbl_reconstructed_image
                )
            
            # Display heatmap
            if ae_data.get('heatmap') is not None:
                display_tensor_image(
                    ae_data['heatmap'],
                    self.image_viewer.lbl_heatmap_image
                )
        
        # Update CNN results
        if result.get('cnn'):
            cnn_data = result['cnn']
            self.results_panel.update_cnn_results(cnn_data)
            
            # Display GradCAM
            if cnn_data.get('gradcam_overlay') is not None:
                display_numpy_image(
                    cnn_data['gradcam_overlay'],
                    self.image_viewer.lbl_gradcam_image
                )
        
        # Update final decision
        final_text, final_color = self.get_final_decision(result)
        self.results_panel.update_final_decision(final_text, final_color)
        
        # Update processing time
        if result.get('processing_time'):
            self.results_panel.update_processing_time(result['processing_time'])
        
        # Update combined view
        self.update_defect_detection_view(result)
        
        # Update LLM chat context
        if self.results_panel.llm_chat:
            self.results_panel.llm_chat.set_context(result)
    
    def update_defect_detection_view(self, result):
        """Update combined detection view"""
        try:
            combined, summary = create_defect_detection_view(self.current_image_path, result)
            if combined is not None:
                display_numpy_image(combined, self.image_viewer.lbl_defect_detection)
        except Exception as e:
            self.results_panel.log_message(f"Error updating detection view: {str(e)}")
    
    def get_final_decision(self, result):
        """Determine final detection decision"""
        # If only CNN is used
        if result['cnn'] and not result['autoencoder']:
            pred = result['cnn']['prediction']
            conf = result['cnn']['confidence']
            if pred == 'ND':
                return "NO DEFECT (NORMAL)", "#44ff44"
            else:
                return f"DEFECT DETECTED: {pred} ({conf:.1%})", "#ff4444"
        
        # Hybrid mode
        if result['cnn'] and result['autoencoder']:
            pred = result['cnn']['prediction']
            conf = result['cnn']['confidence']
            is_anomaly = result['autoencoder']['is_anomaly']
            
            # If CNN is confident (>49%), trust the CNN
            if conf > 0.49:
                if pred == 'ND' and not is_anomaly:
                    return "NO DEFECT (NORMAL)", "#44ff44"
                elif pred == 'ND' and is_anomaly:
                    return f"NO DEFECT (ND: {conf:.1%}) - AE Flagged", "#ffaa00"
                else:
                    return f"DEFECT DETECTED: {pred} ({conf:.1%})", "#ff4444"
            else:
                # CNN uncertain, use autoencoder
                if is_anomaly:
                    return f"DEFECT DETECTED (Uncertain: {pred} {conf:.1%})", "#ff4444"
                else:
                    return "NO DEFECT (LOW CONFIDENCE)", "#ffaa00"
        
        # Only autoencoder
        if result['autoencoder'] and result['autoencoder']['is_anomaly']:
            return "ANOMALY DETECTED", "#ff4444"
        
        return "NO DEFECT (NORMAL)", "#44ff44"
    
    def analyze_all_images(self):
        """Analyze all loaded images in batch"""
        if not self.image_list:
            QMessageBox.warning(self, "Warning", "No images loaded. Please load a folder first.")
            return
        
        if not self.engine:
            QMessageBox.warning(self, "Warning", "Model not loaded!")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Batch Analysis",
            f"Analyze all {len(self.image_list)} images?\nThis may take a while.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Disable buttons
        self.control_panel.btn_analyze_all.setEnabled(False)
        self.control_panel.btn_analyze.setEnabled(False)
        self.control_panel.btn_prev.setEnabled(False)
        self.control_panel.btn_next.setEnabled(False)
        self.control_panel.btn_pause.setEnabled(True)
        self.is_paused = False
        self.control_panel.btn_pause.setText("Pause")
        
        # Track results
        results_summary = []
        total_images = len(self.image_list)
        defects_found = 0
        correct_predictions = 0
        
        # For confusion matrices
        cnn_predictions = {'y_true': [], 'y_pred': []}
        hybrid_predictions = {'y_true': [], 'y_pred': []}
        
        # Process each image
        for i, image_path in enumerate(self.image_list):
            self.current_image_index = i
            self.load_current_image()
            self.control_panel.lbl_image_counter.setText(f"Processing {i + 1} / {total_images}")
            QApplication.processEvents()
            
            try:
                img_path = Path(image_path)
                ground_truth = img_path.parent.name if img_path.parent.name in ['CR', 'LP', 'ND', 'PO'] else None
                
                result = self.engine.predict(image_path, self.config)
                
                if result:
                    self.batch_results_cache[image_path] = result
                    self.display_results(result)
                    QApplication.processEvents()
                    
                    predicted_class = 'Unknown'
                    confidence = 0
                    is_anomaly = False
                    
                    if result.get('cnn'):
                        predicted_class = result['cnn'].get('prediction', 'Unknown')
                        confidence = result['cnn'].get('confidence', 0)
                    
                    if result.get('autoencoder'):
                        is_anomaly = result['autoencoder'].get('is_anomaly', False)
                    
                    # Get hybrid prediction
                    from utils import get_hybrid_prediction
                    hybrid_pred = get_hybrid_prediction(predicted_class, confidence, is_anomaly)
                    
                    if predicted_class != 'ND':
                        defects_found += 1
                    
                    if ground_truth:
                        # Store for confusion matrix (only if ground truth available)
                        cnn_predictions['y_true'].append(ground_truth)
                        cnn_predictions['y_pred'].append(predicted_class)
                        
                        hybrid_predictions['y_true'].append(ground_truth)
                        hybrid_predictions['y_pred'].append(hybrid_pred)
                        
                        if predicted_class == ground_truth:
                            correct_predictions += 1
                    
                    results_summary.append({
                        'file': img_path.name,
                        'ground_truth': ground_truth,
                        'predicted': predicted_class,
                        'hybrid_predicted': hybrid_pred,
                        'confidence': confidence,
                        'correct': predicted_class == ground_truth if ground_truth else None
                    })
                    
                    self.results_panel.log_message(
                        f"[{i+1}/{total_images}] {img_path.name}: {predicted_class} ({confidence:.1%})"
                    )
                    
                    # Check for pause
                    while self.is_paused:
                        QApplication.processEvents()
                        time.sleep(0.1)
                    
                    # Apply delay
                    if self.processing_delay > 0:
                        time.sleep(self.processing_delay)
                    
            except Exception as e:
                self.results_panel.log_message(f"Error processing {img_path.name}: {str(e)}")
        
        # Update statistics
        self.control_panel.lbl_total_processed.setText(f"Processed: {total_images}")
        self.control_panel.lbl_defects_found.setText(f"Defects: {defects_found}")
        
        # Show completion
        images_with_gt = len([r for r in results_summary if r['ground_truth'] is not None])
        if images_with_gt > 0:
            accuracy = correct_predictions / images_with_gt
            self.results_panel.log_message("\n=== Batch Analysis Complete ===")
            self.results_panel.log_message(
                f"Total: {total_images} | Defects: {defects_found} | Accuracy: {accuracy:.1%}"
            )
            
            # Generate and display confusion matrices
            from utils import calculate_metrics
            
            cnn_metrics = calculate_metrics(
                cnn_predictions['y_true'],
                cnn_predictions['y_pred'],
                ['CR', 'LP', 'ND', 'PO']
            )
            
            hybrid_metrics = calculate_metrics(
                hybrid_predictions['y_true'],
                hybrid_predictions['y_pred'],
                ['CR', 'LP', 'ND', 'PO']
            )
            
            cnn_data = {
                'y_true': cnn_predictions['y_true'],
                'y_pred': cnn_predictions['y_pred'],
                'metrics': cnn_metrics
            }
            
            hybrid_data = {
                'y_true': hybrid_predictions['y_true'],
                'y_pred': hybrid_predictions['y_pred'],
                'metrics': hybrid_metrics
            }
            
            self.results_panel.display_confusion_matrices(cnn_data, hybrid_data)
            self.results_panel.log_message(
                f"CNN Accuracy: {cnn_metrics['accuracy']:.2%} | "
                f"Hybrid Accuracy: {hybrid_metrics['accuracy']:.2%}"
            )
            
            # Update LLM chat with batch results and metrics
            if self.results_panel.llm_chat:
                batch_context = {
                    "batch_analysis": True,
                    "total_images": total_images,
                    "defects_found": defects_found,
                    "accuracy": accuracy,
                    "cnn_accuracy": cnn_metrics['accuracy'],
                    "hybrid_accuracy": hybrid_metrics['accuracy'],
                    "images_with_ground_truth": images_with_gt,
                    "cnn_metrics": cnn_metrics,
                    "hybrid_metrics": hybrid_metrics,
                    "class_distribution": {
                        "CR": cnn_predictions['y_true'].count(0) if 0 in cnn_predictions['y_true'] else 0,
                        "LP": cnn_predictions['y_true'].count(1) if 1 in cnn_predictions['y_true'] else 0,
                        "ND": cnn_predictions['y_true'].count(2) if 2 in cnn_predictions['y_true'] else 0,
                        "PO": cnn_predictions['y_true'].count(3) if 3 in cnn_predictions['y_true'] else 0
                    }
                }
                self.results_panel.llm_chat.set_context(batch_context)
                
                # Add confusion matrix to knowledge base (if available)
                try:
                    from sklearn.metrics import confusion_matrix
                    import numpy as np
                    cm = confusion_matrix(
                        cnn_predictions['y_true'],
                        cnn_predictions['y_pred'],
                        labels=[0, 1, 2, 3]  # CR, LP, ND, PO
                    )
                    self.results_panel.llm_chat.add_metrics_to_kb(
                        cm,
                        ['CR', 'LP', 'ND', 'PO'],
                        {
                            "model": "CNN Classifier",
                            "total_images": total_images,
                            "accuracy": cnn_metrics['accuracy'],
                            "date": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    )
                except Exception as e:
                    print(f"Could not add metrics to KB: {e}")
            
            QMessageBox.information(
                self, "Batch Analysis Complete",
                f"Processed: {total_images} images\n"
                f"Defects Found: {defects_found}\n"
                f"CNN Accuracy: {cnn_metrics['accuracy']:.2%}\n"
                f"Hybrid Accuracy: {hybrid_metrics['accuracy']:.2%}\n\n"
                f"View confusion matrices in the 'Metrics' tab"
            )
        else:
            self.results_panel.log_message("\n=== Batch Analysis Complete ===")
            self.results_panel.log_message(f"Total: {total_images} | Defects: {defects_found}")
            QMessageBox.information(
                self, "Batch Analysis Complete",
                f"Processed: {total_images} images\nDefects Found: {defects_found}"
            )
        
        self.batch_results = results_summary
        
        # Re-enable buttons and review mode
        self.control_panel.btn_analyze_all.setEnabled(True)
        self.control_panel.btn_analyze.setEnabled(True)
        self.control_panel.combo_review_mode.setEnabled(True)
        self.control_panel.btn_pause.setEnabled(False)
        self.control_panel.btn_pause.setText("Pause")
        self.is_paused = False
        self.update_navigation_buttons()
    
    def export_results(self):
        """Export results to JSON file"""
        # Check if we have any results - either from cache or current result
        if not self.current_result and not self.batch_results_cache and not self.batch_results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        
        try:
            from datetime import datetime
            
            # Ask user where to save
            default_filename = f"weld_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Results",
                default_filename,
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Prepare export data
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "system_info": {
                    "cnn_model": str(Path(self.config.cnn_model_path).name) if hasattr(self.config, 'cnn_model_path') else "best_model_pytorch.pth",
                    "cae_model": str(Path(self.config.autoencoder_model_path).name) if hasattr(self.config, 'autoencoder_model_path') else "best_model.pth",
                    "detection_mode": getattr(self.config, 'detection_mode', 'hybrid'),
                    "cae_threshold": self.config.ae_threshold if hasattr(self.config, 'ae_threshold') else 0.018
                },
                "results": []
            }
            
            # Add batch results - check both cache and batch_results list
            results_to_export = []
            
            if self.batch_results_cache:
                # Use cache (has full result data)
                results_to_export = list(self.batch_results_cache.values())
            elif self.batch_results:
                # Use batch_results list
                results_to_export = self.batch_results
            elif self.current_result:
                # Single result
                results_to_export = [self.current_result]
            
            if results_to_export:
                for result in results_to_export:
                    export_data["results"].append(self._serialize_result(result))
                
                # Add batch statistics
                export_data["batch_statistics"] = {
                    "total_images": len(results_to_export),
                    "defects_found": sum(1 for r in results_to_export if r.get('final_decision') != 'ND'),
                    "class_distribution": {},
                    "average_confidence": 0,
                    "average_processing_time": 0
                }
                
                # Calculate statistics
                class_counts = {}
                total_confidence = 0
                total_time = 0
                
                for result in results_to_export:
                    # Class distribution
                    decision = result.get('final_decision', 'Unknown')
                    class_counts[decision] = class_counts.get(decision, 0) + 1
                    
                    # Average confidence
                    cnn_conf = result.get('cnn', {}).get('confidence', 0)
                    total_confidence += cnn_conf
                    
                    # Average time
                    total_time += result.get('processing_time', 0)
                
                export_data["batch_statistics"]["class_distribution"] = class_counts
                export_data["batch_statistics"]["average_confidence"] = total_confidence / len(results_to_export) if results_to_export else 0
                export_data["batch_statistics"]["average_processing_time"] = total_time / len(results_to_export) if results_to_export else 0
            
            # Add current single result if no batch results
            elif self.current_result:
                export_data["results"].append(self._serialize_result(self.current_result))
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            # Add to knowledge base for LLM
            try:
                if hasattr(self.results_panel, 'llm_chat') and self.results_panel.llm_chat:
                    # Create summary document
                    summary = self._create_results_summary(export_data)
                    self.results_panel.llm_chat.knowledge_base.add_document(
                        summary,
                        metadata={
                            "title": f"Detection Results {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            "category": "detection_results",
                            "source": "export",
                            "file_path": file_path
                        }
                    )
                    self.results_panel.log_message(f"✓ Results added to LLM knowledge base")
            except Exception as e:
                print(f"Warning: Could not add to knowledge base: {e}")
            
            self.results_panel.log_message(f"✓ Results exported to: {file_path}")
            QMessageBox.information(
                self, 
                "Export Successful", 
                f"Results exported successfully!\n\n{len(export_data['results'])} detections saved to:\n{Path(file_path).name}"
            )
            
        except Exception as e:
            self.results_panel.log_message(f"✗ Export failed: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")
    
    def _serialize_result(self, result):
        """Convert result dict to JSON-serializable format"""
        serialized = {
            "image_path": result.get('image_path', ''),
            "image_name": Path(result.get('image_path', '')).name if result.get('image_path') else '',
            "timestamp": result.get('timestamp', ''),
            "processing_time": result.get('processing_time', 0),
            "final_decision": result.get('final_decision', 'Unknown')
        }
        
        # Autoencoder results
        if result.get('autoencoder'):
            ae = result['autoencoder']
            serialized['autoencoder'] = {
                "is_anomaly": ae.get('is_anomaly', False),
                "reconstruction_error": float(ae.get('reconstruction_error', 0)),
                "threshold": float(ae.get('threshold', 0)),
                "confidence": float(ae.get('confidence', 0))
            }
        
        # CNN results
        if result.get('cnn'):
            cnn = result['cnn']
            serialized['cnn'] = {
                "prediction": cnn.get('prediction', 'Unknown'),
                "confidence": float(cnn.get('confidence', 0)),
                "probabilities": {k: float(v) for k, v in cnn.get('probabilities', {}).items()}
            }
        
        return serialized
    
    def _create_results_summary(self, export_data):
        """Create text summary for knowledge base"""
        summary = f"# Detection Results Summary\n\n"
        summary += f"**Export Date:** {export_data['export_timestamp']}\n\n"
        
        if export_data.get('batch_statistics'):
            stats = export_data['batch_statistics']
            summary += f"## Batch Analysis Statistics\n\n"
            summary += f"- **Total Images Processed:** {stats['total_images']}\n"
            summary += f"- **Defects Found:** {stats['defects_found']}\n"
            summary += f"- **Average Confidence:** {stats['average_confidence']:.2%}\n"
            summary += f"- **Average Processing Time:** {stats['average_processing_time']:.3f}s\n\n"
            
            summary += f"### Class Distribution\n"
            for class_name, count in stats['class_distribution'].items():
                percentage = (count / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
                summary += f"- **{class_name}:** {count} images ({percentage:.1f}%)\n"
            summary += "\n"
        
        # Add sample results
        summary += f"## Individual Detection Results\n\n"
        for i, result in enumerate(export_data['results'][:10], 1):  # First 10 results
            summary += f"### Image {i}: {result['image_name']}\n"
            summary += f"- **Final Decision:** {result['final_decision']}\n"
            
            if result.get('cnn'):
                cnn = result['cnn']
                summary += f"- **CNN Prediction:** {cnn['prediction']} (confidence: {cnn['confidence']:.2%})\n"
            
            if result.get('autoencoder'):
                ae = result['autoencoder']
                summary += f"- **Autoencoder:** {'Anomaly detected' if ae['is_anomaly'] else 'Normal'} "
                summary += f"(error: {ae['reconstruction_error']:.4f})\n"
            
            summary += f"- **Processing Time:** {result['processing_time']:.3f}s\n\n"
        
        if len(export_data['results']) > 10:
            summary += f"\n*... and {len(export_data['results']) - 10} more results*\n"
        
        return summary
    
    def clear_results(self):
        """Clear current results"""
        self.image_viewer.lbl_original_image.clear()
        self.image_viewer.lbl_original_image.setText("No Image Loaded")
        self.image_viewer.clear_results()
        self.results_panel.clear_results()
    
    def clear_all_results(self):
        """Clear all results and reset"""
        self.clear_results()
        self.batch_results_cache = {}
        self.batch_results = []
        self.control_panel.lbl_total_processed.setText("Processed: 0")
        self.control_panel.lbl_defects_found.setText("Defects: 0")
        self.control_panel.lbl_avg_time.setText("Avg Time: 0.00s")
        self.results_panel.log_message("All results cleared")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About",
            "Weld Defect Detection System v1.0\n\n"
            "A professional system combining CNN classification\n"
            "and Convolutional Autoencoder for defect detection.\n\n"
            "Detects: Cracks (CR), Lack of Penetration (LP),\n"
            "No Defect (ND), Porosity (PO)"
        )


def main():
    app = QApplication(sys.argv)
    window = WeldDefectGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
