"""
Professional Weld Defect Detection System - Main GUI
Combines CNN Classifier and Autoencoder for comprehensive defect detection
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QGroupBox, QSlider, QCheckBox, QComboBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QSplitter, QStatusBar, QMenuBar,
    QAction, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont

from inference_engine import DefectDetectionEngine
from config import GUIConfig
from llm_chat import LLMChatWidget
import time

class ImageProcessingThread(QThread):
    """Background thread for image processing"""
    progress_updated = pyqtSignal(int, str)
    result_ready = pyqtSignal(dict)
    finished = pyqtSignal()
    
    def __init__(self, engine, image_path, config):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.config = config
        
    def run(self):
        try:
            result = self.engine.predict(self.image_path, self.config)
            self.result_ready.emit(result)
        except Exception as e:
            self.progress_updated.emit(0, f"Error: {str(e)}")
        finally:
            self.finished.emit()


class WeldDefectGUI(QMainWindow):
    """Main GUI window"""
    
    def __init__(self):
        super().__init__()
        self.config = GUIConfig()
        self.engine = None
        self.current_image_path = None
        self.current_result = None
        self.processing_thread = None
        self.llm_chat = None
        
        # Batch processing
        self.image_list = []
        self.current_image_index = 0
        self.batch_results_cache = {}  # Store results for each image path
        
        self.initUI()
        self.load_models()
        
    def initUI(self):
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
        
        # Left panel - Controls
        left_panel = self.create_left_panel()
        
        # Center panel - Image display
        center_panel = self.create_center_panel()
        
        # Right panel - Results
        right_panel = self.create_right_panel()
        
        # Add panels to splitter for resizable layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700, 400])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
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
        
        export_action = QAction('Export Results', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu('Settings')
        
        theme_action = QAction('Toggle Theme', self)
        theme_action.triggered.connect(self.toggle_theme)
        settings_menu.addAction(theme_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("CONTROL PANEL")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Upload section
        upload_group = QGroupBox("Input")
        upload_layout = QVBoxLayout()
        
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_image.setMinimumHeight(40)
        upload_layout.addWidget(self.btn_load_image)
        
        self.btn_load_folder = QPushButton("Load Folder")
        self.btn_load_folder.clicked.connect(self.load_folder)
        self.btn_load_folder.setMinimumHeight(40)
        upload_layout.addWidget(self.btn_load_folder)
        
        # Navigation for batch processing
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setEnabled(False)
        nav_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_next)
        upload_layout.addLayout(nav_layout)
        
        # Image counter label
        self.lbl_image_counter = QLabel("No images loaded")
        self.lbl_image_counter.setAlignment(Qt.AlignCenter)
        self.lbl_image_counter.setStyleSheet("color: #888; font-size: 10px;")
        upload_layout.addWidget(self.lbl_image_counter)
        
        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)
        
        # Detection Mode section
        mode_group = QGroupBox("Detection Mode")
        mode_layout = QVBoxLayout()
        
        self.combo_detection_mode = QComboBox()
        self.combo_detection_mode.addItems([
            "Supervised (CNN)",
            "Unsupervised (Autoencoder)", 
            "Hybrid (Both)"
        ])
        self.combo_detection_mode.currentIndexChanged.connect(self.on_detection_mode_changed)
        self.combo_detection_mode.setMinimumHeight(35)
        mode_layout.addWidget(self.combo_detection_mode)
        
        # Mode description
        self.lbl_mode_desc = QLabel("CNN classification with GradCAM visualization")
        self.lbl_mode_desc.setWordWrap(True)
        self.lbl_mode_desc.setStyleSheet("color: #888; font-size: 10px; padding: 5px;")
        mode_layout.addWidget(self.lbl_mode_desc)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        
        self.btn_analyze = QPushButton("Analyze Image")
        self.btn_analyze.clicked.connect(self.analyze_image)
        self.btn_analyze.setMinimumHeight(50)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet("font-size: 14px; font-weight: bold;")
        action_layout.addWidget(self.btn_analyze)
        
        self.btn_analyze_all = QPushButton("Analyze All")
        self.btn_analyze_all.clicked.connect(self.analyze_all_images)
        self.btn_analyze_all.setMinimumHeight(45)
        self.btn_analyze_all.setEnabled(False)
        self.btn_analyze_all.setStyleSheet("font-size: 12px;")
        action_layout.addWidget(self.btn_analyze_all)
        
        self.btn_export = QPushButton("Export Results")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setMinimumHeight(40)
        action_layout.addWidget(self.btn_export)
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_results)
        self.btn_clear.setMinimumHeight(40)
        action_layout.addWidget(self.btn_clear)
        
        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.clear_all_results)
        self.btn_clear_all.setMinimumHeight(40)
        self.btn_clear_all.setStyleSheet("color: #ff6666;")
        action_layout.addWidget(self.btn_clear_all)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.lbl_total_processed = QLabel("Processed: 0")
        stats_layout.addWidget(self.lbl_total_processed)
        
        self.lbl_defects_found = QLabel("Defects: 0")
        stats_layout.addWidget(self.lbl_defects_found)
        
        self.lbl_avg_time = QLabel("Avg Time: 0.00s")
        stats_layout.addWidget(self.lbl_avg_time)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
        
    def create_center_panel(self):
        """Create center image display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tabs for different views - organized by detection mode
        self.tab_widget = QTabWidget()
        
        # Common image size for all tabs
        IMG_SIZE = 580
        
        # ===== INPUT TAB =====
        original_tab = QWidget()
        original_layout = QVBoxLayout(original_tab)
        original_layout.setContentsMargins(0, 5, 0, 5)
        original_layout.setSpacing(5)
        
        # Ground truth label (above image)
        self.lbl_ground_truth = QLabel("Ground Truth: Unknown")
        self.lbl_ground_truth.setAlignment(Qt.AlignCenter)
        self.lbl_ground_truth.setStyleSheet("""
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 8px;
            font-size: 13px;
            font-weight: bold;
        """)
        self.lbl_ground_truth.setFixedHeight(35)
        original_layout.addWidget(self.lbl_ground_truth)
        
        self.lbl_original_image = QLabel()
        self.lbl_original_image.setAlignment(Qt.AlignCenter)
        self.lbl_original_image.setFixedSize(IMG_SIZE, IMG_SIZE - 70)
        self.lbl_original_image.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.lbl_original_image.setText("No Image Loaded")
        original_layout.addWidget(self.lbl_original_image, alignment=Qt.AlignCenter)
        
        # Info label for Input tab
        self.lbl_info_input = QLabel("Original input image - Load an image or folder to begin analysis")
        self.lbl_info_input.setAlignment(Qt.AlignCenter)
        self.lbl_info_input.setStyleSheet("color: #aaa; font-size: 20px; padding: 5px;")
        original_layout.addWidget(self.lbl_info_input)
        self.tab_widget.addTab(original_tab, "Input")
        
        # ===== SUPERVISED (CNN) TAB =====
        gradcam_tab = QWidget()
        gradcam_layout = QVBoxLayout(gradcam_tab)
        gradcam_layout.setContentsMargins(0, 5, 0, 5)
        gradcam_layout.setSpacing(5)
        
        # Ground truth label for GradCAM tab
        self.lbl_ground_truth_gradcam = QLabel("Ground Truth: Unknown")
        self.lbl_ground_truth_gradcam.setAlignment(Qt.AlignCenter)
        self.lbl_ground_truth_gradcam.setStyleSheet("""
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 8px;
            font-size: 13px;
            font-weight: bold;
        """)
        self.lbl_ground_truth_gradcam.setFixedHeight(35)
        gradcam_layout.addWidget(self.lbl_ground_truth_gradcam)
        
        self.lbl_gradcam_image = QLabel()
        self.lbl_gradcam_image.setAlignment(Qt.AlignCenter)
        self.lbl_gradcam_image.setFixedSize(IMG_SIZE, IMG_SIZE - 70)
        self.lbl_gradcam_image.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.lbl_gradcam_image.setText("CNN GradCAM visualization\n(Supervised mode)")
        gradcam_layout.addWidget(self.lbl_gradcam_image, alignment=Qt.AlignCenter)
        
        # Info label for GradCAM tab
        self.lbl_info_gradcam = QLabel("GradCAM highlights regions the CNN uses to make predictions - Red/yellow areas indicate high importance")
        self.lbl_info_gradcam.setAlignment(Qt.AlignCenter)
        self.lbl_info_gradcam.setStyleSheet("color: #aaa; font-size: 20px; padding: 5px;")
        self.lbl_info_gradcam.setWordWrap(True)
        gradcam_layout.addWidget(self.lbl_info_gradcam)
        self.tab_widget.addTab(gradcam_tab, "CNN GradCAM")
        
        # ===== UNSUPERVISED (AUTOENCODER) TABS =====
        # Reconstructed image tab
        recon_tab = QWidget()
        recon_layout = QVBoxLayout(recon_tab)
        recon_layout.setContentsMargins(0, 5, 0, 5)
        recon_layout.setSpacing(5)
        
        # Ground truth label for Reconstruction tab
        self.lbl_ground_truth_recon = QLabel("Ground Truth: Unknown")
        self.lbl_ground_truth_recon.setAlignment(Qt.AlignCenter)
        self.lbl_ground_truth_recon.setStyleSheet("""
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 8px;
            font-size: 13px;
            font-weight: bold;
        """)
        self.lbl_ground_truth_recon.setFixedHeight(35)
        recon_layout.addWidget(self.lbl_ground_truth_recon)
        
        self.lbl_reconstructed_image = QLabel()
        self.lbl_reconstructed_image.setAlignment(Qt.AlignCenter)
        self.lbl_reconstructed_image.setFixedSize(IMG_SIZE, IMG_SIZE - 70)
        self.lbl_reconstructed_image.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.lbl_reconstructed_image.setText("Autoencoder reconstruction\n(Unsupervised mode)")
        recon_layout.addWidget(self.lbl_reconstructed_image, alignment=Qt.AlignCenter)
        
        # Info label for Reconstruction tab
        self.lbl_info_recon = QLabel("Autoencoder attempts to reconstruct the input - Poor reconstruction indicates anomaly/defect")
        self.lbl_info_recon.setAlignment(Qt.AlignCenter)
        self.lbl_info_recon.setStyleSheet("color: #aaa; font-size: 20px; padding: 5px;")
        self.lbl_info_recon.setWordWrap(True)
        recon_layout.addWidget(self.lbl_info_recon)
        self.tab_widget.addTab(recon_tab, "AE Reconstruction")
        
        # Anomaly Heatmap tab
        heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_tab)
        heatmap_layout.setContentsMargins(0, 5, 0, 5)
        heatmap_layout.setSpacing(5)
        
        # Ground truth label for Anomaly Map tab
        self.lbl_ground_truth_heatmap = QLabel("Ground Truth: Unknown")
        self.lbl_ground_truth_heatmap.setAlignment(Qt.AlignCenter)
        self.lbl_ground_truth_heatmap.setStyleSheet("""
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 8px;
            font-size: 13px;
            font-weight: bold;
        """)
        self.lbl_ground_truth_heatmap.setFixedHeight(35)
        heatmap_layout.addWidget(self.lbl_ground_truth_heatmap)
        
        self.lbl_heatmap_image = QLabel()
        self.lbl_heatmap_image.setAlignment(Qt.AlignCenter)
        self.lbl_heatmap_image.setFixedSize(IMG_SIZE, IMG_SIZE - 70)
        self.lbl_heatmap_image.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.lbl_heatmap_image.setText("Anomaly heatmap\n(Unsupervised mode)")
        heatmap_layout.addWidget(self.lbl_heatmap_image, alignment=Qt.AlignCenter)
        
        # Info label for Anomaly Map tab
        self.lbl_info_heatmap = QLabel("Pixel-wise reconstruction error - Bright areas show where the autoencoder failed to reconstruct (potential defects)")
        self.lbl_info_heatmap.setAlignment(Qt.AlignCenter)
        self.lbl_info_heatmap.setStyleSheet("color: #aaa; font-size: 20px; padding: 5px;")
        self.lbl_info_heatmap.setWordWrap(True)
        heatmap_layout.addWidget(self.lbl_info_heatmap)
        self.tab_widget.addTab(heatmap_tab, "AE Anomaly Map")
        
        # ===== RESULTS TAB =====
        defect_tab = QWidget()
        defect_layout = QVBoxLayout(defect_tab)
        defect_layout.setContentsMargins(5, 5, 5, 5)
        defect_layout.setSpacing(5)
        
        # Detection results summary section (above image)
        self.lbl_results_summary = QLabel("Detection Results")
        self.lbl_results_summary.setAlignment(Qt.AlignLeft)
        self.lbl_results_summary.setStyleSheet("""
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 8px;
            font-size: 12px;
        """)
        self.lbl_results_summary.setFixedHeight(100)
        defect_layout.addWidget(self.lbl_results_summary)
        
        # Combined visualization image
        self.lbl_defect_detection = QLabel()
        self.lbl_defect_detection.setAlignment(Qt.AlignCenter)
        self.lbl_defect_detection.setFixedSize(IMG_SIZE, IMG_SIZE - 130)
        self.lbl_defect_detection.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.lbl_defect_detection.setText("Combined detection results")
        defect_layout.addWidget(self.lbl_defect_detection, alignment=Qt.AlignCenter)
        
        # Info label for Results tab
        self.lbl_info_results = QLabel("Combined visualization with GradCAM overlay - Green box marks detected anomaly region")
        self.lbl_info_results.setAlignment(Qt.AlignCenter)
        self.lbl_info_results.setStyleSheet("color: #aaa; font-size: 20px; padding: 5px;")
        self.lbl_info_results.setWordWrap(True)
        defect_layout.addWidget(self.lbl_info_results)
        self.tab_widget.addTab(defect_tab, "Results")
        
        layout.addWidget(self.tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return panel
        
    def create_right_panel(self):
        """Create right results panel with tabs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 10px 20px;
                border: 1px solid #3a3a3a;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #0066cc;
            }
            QTabBar::tab:hover {
                background-color: #3a3a3a;
            }
        """)
        
        # Tab 1: Results
        results_tab = self.create_results_tab()
        tabs.addTab(results_tab, "Results")
        
        # Tab 2: AI Assistant
        self.llm_chat = LLMChatWidget()
        tabs.addTab(self.llm_chat, "AI Assistant")
        
        layout.addWidget(tabs)
        
        return panel
    
    def create_results_tab(self):
        """Create the results tab content"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("DETECTION RESULTS")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Autoencoder results
        ae_group = QGroupBox("Autoencoder Analysis")
        ae_layout = QVBoxLayout()
        
        self.lbl_ae_status = QLabel("Status: Not Analyzed")
        self.lbl_ae_status.setStyleSheet("font-size: 13px; font-weight: bold;")
        ae_layout.addWidget(self.lbl_ae_status)
        
        self.lbl_ae_error = QLabel("Reconstruction Error: -")
        ae_layout.addWidget(self.lbl_ae_error)
        
        self.lbl_ae_threshold = QLabel("Threshold: -")
        ae_layout.addWidget(self.lbl_ae_threshold)
        
        ae_group.setLayout(ae_layout)
        layout.addWidget(ae_group)
        
        # CNN results
        cnn_group = QGroupBox("CNN Classification")
        cnn_layout = QVBoxLayout()
        
        self.lbl_cnn_prediction = QLabel("Prediction: Not Analyzed")
        self.lbl_cnn_prediction.setStyleSheet("font-size: 13px; font-weight: bold;")
        cnn_layout.addWidget(self.lbl_cnn_prediction)
        
        self.lbl_cnn_confidence = QLabel("Confidence: -")
        cnn_layout.addWidget(self.lbl_cnn_confidence)
        
        # Confidence bars
        conf_label = QLabel("Class Probabilities:")
        conf_label.setStyleSheet("margin-top: 10px;")
        cnn_layout.addWidget(conf_label)
        
        self.confidence_bars = {}
        for class_name in ['CR', 'LP', 'ND', 'PO']:
            bar_layout = QHBoxLayout()
            label = QLabel(f"{class_name}:")
            label.setMinimumWidth(30)
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            bar_layout.addWidget(label)
            bar_layout.addWidget(bar)
            cnn_layout.addLayout(bar_layout)
            self.confidence_bars[class_name] = bar
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group)
        
        # Final decision
        decision_group = QGroupBox("Final Decision")
        decision_layout = QVBoxLayout()
        
        self.lbl_final_decision = QLabel("AWAITING ANALYSIS")
        self.lbl_final_decision.setAlignment(Qt.AlignCenter)
        self.lbl_final_decision.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            padding: 20px;
            border: 2px solid #555;
            border-radius: 5px;
        """)
        decision_layout.addWidget(self.lbl_final_decision)
        
        self.lbl_processing_time = QLabel("Processing Time: -")
        self.lbl_processing_time.setAlignment(Qt.AlignCenter)
        decision_layout.addWidget(self.lbl_processing_time)
        
        decision_group.setLayout(decision_layout)
        layout.addWidget(decision_group)
        
        # Log
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout()
        
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(200)
        log_layout.addWidget(self.txt_log)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return panel
        
    def get_stylesheet(self):
        """Get dark theme stylesheet"""
        return """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
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
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #555;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #0e639c;
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
            self.log_message("Loading models...")
            self.engine = DefectDetectionEngine(self.config)
            self.log_message("Models loaded successfully")
            self.status_bar.showMessage("Models loaded - Ready")
        except Exception as e:
            self.log_message(f"✗ Error loading models: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load models:\n{str(e)}")
            
    def load_image(self):
        """Load single image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.btn_analyze.setEnabled(True)
            self.log_message(f"Loaded: {Path(file_path).name}")
            self.status_bar.showMessage(f"Image loaded: {Path(file_path).name}")
            
    def load_folder(self):
        """Load folder for batch processing - supports folders with subclass subfolders"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder (with CR, LP, ND, PO subfolders)")
        
        if folder_path:
            folder = Path(folder_path)
            self.image_list = []
            self.batch_results_cache = {}  # Clear previous batch results
            
            # Supported image extensions
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
            
            # Check for subclass folders (CR, LP, ND, PO)
            subfolders = ['CR', 'LP', 'ND', 'PO']
            has_subfolders = any((folder / sf).exists() for sf in subfolders)
            
            if has_subfolders:
                # Load from subfolders
                for subfolder in subfolders:
                    subfolder_path = folder / subfolder
                    if subfolder_path.exists():
                        for ext in extensions:
                            self.image_list.extend(subfolder_path.glob(ext))
                self.log_message(f"Loaded from subfolders: {', '.join(subfolders)}")
            else:
                # Load directly from folder
                for ext in extensions:
                    self.image_list.extend(folder.glob(ext))
            
            # Convert to strings and sort
            self.image_list = sorted([str(p) for p in self.image_list])
            
            if self.image_list:
                self.current_image_index = 0
                self.load_current_image()
                self.update_navigation_buttons()
                self.btn_analyze_all.setEnabled(True)
                self.log_message(f"Loaded {len(self.image_list)} images from folder")
                self.status_bar.showMessage(f"Loaded {len(self.image_list)} images")
            else:
                QMessageBox.warning(self, "Warning", "No images found in the selected folder.")
                self.log_message("No images found in folder")
    
    def load_current_image(self):
        """Load and display the current image from the list"""
        if self.image_list and 0 <= self.current_image_index < len(self.image_list):
            self.current_image_path = self.image_list[self.current_image_index]
            self.display_image(self.current_image_path)
            self.btn_analyze.setEnabled(True)
            
            # Show filename and class if in subfolder
            img_path = Path(self.current_image_path)
            parent_name = img_path.parent.name
            if parent_name in ['CR', 'LP', 'ND', 'PO']:
                display_name = f"[{parent_name}] {img_path.name}"
                self.update_ground_truth_label(parent_name)
            else:
                display_name = img_path.name
                self.update_ground_truth_label(None)
            self.status_bar.showMessage(f"Image: {display_name}")
    
    def update_navigation_buttons(self):
        """Update navigation buttons and counter based on current state"""
        if self.image_list:
            self.btn_prev.setEnabled(self.current_image_index > 0)
            self.btn_next.setEnabled(self.current_image_index < len(self.image_list) - 1)
            self.lbl_image_counter.setText(f"{self.current_image_index + 1} / {len(self.image_list)}")
        else:
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.lbl_image_counter.setText("No images loaded")
    
    def prev_image(self):
        """Go to previous image"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            self.update_navigation_buttons()
            # Display cached results if available
            self.display_cached_results()
    
    def next_image(self):
        """Go to next image"""
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_current_image()
            self.update_navigation_buttons()
            # Display cached results if available
            self.display_cached_results()
    
    def display_cached_results(self):
        """Display cached results for current image if available"""
        if self.current_image_path and self.current_image_path in self.batch_results_cache:
            result = self.batch_results_cache[self.current_image_path]
            self.display_results(result)
            self.log_message(f"Showing cached results for {Path(self.current_image_path).name}")
        else:
            # Clear only visualization tabs but keep input image
            self.lbl_gradcam_image.clear()
            self.lbl_gradcam_image.setText("Click 'Analyze Image' to process")
            self.lbl_reconstructed_image.clear()
            self.lbl_reconstructed_image.setText("Not analyzed")
            self.lbl_heatmap_image.clear()
            self.lbl_heatmap_image.setText("Not analyzed")
            self.lbl_defect_detection.clear()
            self.lbl_defect_detection.setText("Not analyzed")
            self.lbl_results_summary.setText("No analysis results yet")
    
    def analyze_all_images(self):
        """Analyze all loaded images in batch"""
        if not self.image_list:
            QMessageBox.warning(self, "Warning", "No images loaded. Please load a folder first.")
            return
        
        if not self.engine:
            QMessageBox.warning(self, "Warning", "Model not loaded!")
            return
        
        # Confirm with user
        reply = QMessageBox.question(
            self, "Confirm Batch Analysis",
            f"Analyze all {len(self.image_list)} images?\nThis may take a while.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Disable buttons during batch processing
        self.btn_analyze_all.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        
        # Track results
        results_summary = []
        total_images = len(self.image_list)
        defects_found = 0
        correct_predictions = 0
        
        # Process each image
        for i, image_path in enumerate(self.image_list):
            self.current_image_index = i
            self.load_current_image()
            self.lbl_image_counter.setText(f"Processing {i + 1} / {total_images}")
            QApplication.processEvents()  # Update UI
            
            try:
                # Get ground truth from folder name
                img_path = Path(image_path)
                ground_truth = img_path.parent.name if img_path.parent.name in ['CR', 'LP', 'ND', 'PO'] else None
                
                # Run analysis using predict() method
                result = self.engine.predict(image_path, self.config)
                
                if result:
                    # Cache results for this image
                    self.batch_results_cache[image_path] = result
                    
                    # Display results in tabs (GradCAM, Results, etc.)
                    self.display_results(result)
                    QApplication.processEvents()  # Update UI to show results
                    
                    # Extract prediction info from CNN results
                    predicted_class = 'Unknown'
                    confidence = 0
                    if result.get('cnn'):
                        predicted_class = result['cnn'].get('prediction', 'Unknown')
                        confidence = result['cnn'].get('confidence', 0)
                    
                    # Track defects
                    if predicted_class != 'ND':
                        defects_found += 1
                    
                    # Track accuracy if ground truth available
                    if ground_truth:
                        if predicted_class == ground_truth:
                            correct_predictions += 1
                    
                    results_summary.append({
                        'file': img_path.name,
                        'ground_truth': ground_truth,
                        'predicted': predicted_class,
                        'confidence': confidence,
                        'correct': predicted_class == ground_truth if ground_truth else None
                    })
                    
                    self.log_message(f"[{i+1}/{total_images}] {img_path.name}: {predicted_class} ({confidence:.1%})")
                    
            except Exception as e:
                self.log_message(f"Error processing {img_path.name}: {str(e)}")
                results_summary.append({
                    'file': img_path.name,
                    'ground_truth': ground_truth,
                    'predicted': 'ERROR',
                    'confidence': 0,
                    'correct': False
                })
        
        # Update statistics
        self.lbl_total_processed.setText(f"Processed: {total_images}")
        self.lbl_defects_found.setText(f"Defects: {defects_found}")
        
        # Calculate and show accuracy if ground truth was available
        images_with_gt = len([r for r in results_summary if r['ground_truth'] is not None])
        if images_with_gt > 0:
            accuracy = correct_predictions / images_with_gt
            self.log_message(f"\n=== Batch Analysis Complete ===")
            self.log_message(f"Total: {total_images} | Defects: {defects_found} | Accuracy: {accuracy:.1%}")
            QMessageBox.information(
                self, "Batch Analysis Complete",
                f"Processed: {total_images} images\n"
                f"Defects Found: {defects_found}\n"
                f"Accuracy: {accuracy:.1%} ({correct_predictions}/{images_with_gt})"
            )
        else:
            self.log_message(f"\n=== Batch Analysis Complete ===")
            self.log_message(f"Total: {total_images} | Defects: {defects_found}")
            QMessageBox.information(
                self, "Batch Analysis Complete",
                f"Processed: {total_images} images\n"
                f"Defects Found: {defects_found}"
            )
        
        # Store batch results for export
        self.batch_results = results_summary
        
        # Re-enable buttons
        self.btn_analyze_all.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.update_navigation_buttons()
            
    def display_image(self, image_path):
        """Display image in GUI"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            self.lbl_original_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.lbl_original_image.setPixmap(scaled_pixmap)
        
        # Try to detect ground truth from path
        img_path = Path(image_path)
        parent_name = img_path.parent.name
        if parent_name in ['CR', 'LP', 'ND', 'PO']:
            self.update_ground_truth_label(parent_name)
        else:
            self.update_ground_truth_label(None)
    
    def update_ground_truth_label(self, class_name):
        """Update the ground truth labels across all tabs"""
        class_full_names = {
            'CR': 'Crack',
            'LP': 'Lack of Penetration',
            'ND': 'No Defect',
            'PO': 'Porosity'
        }
        
        # List of all ground truth labels
        gt_labels = [
            self.lbl_ground_truth,           # Input tab
            self.lbl_ground_truth_gradcam,   # CNN GradCAM tab
            self.lbl_ground_truth_recon,     # AE Reconstruction tab
            self.lbl_ground_truth_heatmap,   # AE Anomaly Map tab
        ]
        
        if class_name and class_name in class_full_names:
            full_name = class_full_names[class_name]
            if class_name == 'ND':
                color = "#44ff44"  # Green for no defect
            else:
                color = "#ff9944"  # Orange for defects
            
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
        
        # Update all ground truth labels
        for label in gt_labels:
            label.setText(text)
            label.setStyleSheet(style)
    
    def on_detection_mode_changed(self, index):
        """Handle detection mode change"""
        mode_descriptions = [
            "CNN classification with GradCAM visualization.\nClassifies defects into: CR, LP, ND, PO",
            "Autoencoder anomaly detection.\nDetects anomalies by reconstruction error",
            "Combined CNN + Autoencoder.\nAutoencoder screens for anomalies, CNN classifies defects"
        ]
        self.lbl_mode_desc.setText(mode_descriptions[index])
        
        # Update config based on mode
        if index == 0:  # Supervised (CNN)
            self.config.use_cnn = True
            self.config.use_autoencoder = False
        elif index == 1:  # Unsupervised (Autoencoder)
            self.config.use_cnn = False
            self.config.use_autoencoder = True
        else:  # Hybrid
            self.config.use_cnn = True
            self.config.use_autoencoder = True
        
        self.log_message(f"Detection mode: {self.combo_detection_mode.currentText()}")
        
    def analyze_image(self):
        """Analyze current image"""
        if not self.current_image_path:
            return
            
        if self.processing_thread and self.processing_thread.isRunning():
            return
        
        # Get current detection mode
        mode_index = self.combo_detection_mode.currentIndex()
        mode_name = ["Supervised (CNN)", "Unsupervised (Autoencoder)", "Hybrid"][mode_index]
        
        self.log_message(f"Starting {mode_name} analysis...")
        self.btn_analyze.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Set detection mode based on combo selection
        if mode_index == 0:  # Supervised (CNN)
            self.config.use_cnn = True
            self.config.use_autoencoder = False
        elif mode_index == 1:  # Unsupervised (Autoencoder)
            self.config.use_cnn = False
            self.config.use_autoencoder = True
        else:  # Hybrid
            self.config.use_cnn = True
            self.config.use_autoencoder = True
        
        # Start processing in background thread
        self.processing_thread = ImageProcessingThread(
            self.engine, self.current_image_path, self.config
        )
        self.processing_thread.result_ready.connect(self.display_results)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()
        
    def display_results(self, result):
        """Display analysis results"""
        self.current_result = result
        
        # Autoencoder results
        if result['autoencoder']:
            ae_data = result['autoencoder']
            status_text = "ANOMALY DETECTED" if ae_data['is_anomaly'] else "NORMAL"
            status_color = "#ff4444" if ae_data['is_anomaly'] else "#44ff44"
            
            self.lbl_ae_status.setText(f"Status: {status_text}")
            self.lbl_ae_status.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {status_color};")
            self.lbl_ae_error.setText(f"Reconstruction Error: {ae_data['error']:.6f}")
            self.lbl_ae_threshold.setText(f"Threshold: {ae_data['threshold']:.6f}")
            
            # Display reconstructed image
            if 'reconstruction' in ae_data:
                self.display_tensor_image(ae_data['reconstruction'], self.lbl_reconstructed_image)
                
            # Display heatmap
            if 'heatmap' in ae_data:
                self.display_tensor_image(ae_data['heatmap'], self.lbl_heatmap_image)
        
        # CNN results
        if result['cnn']:
            cnn_data = result['cnn']
            pred_class = cnn_data['prediction']
            confidence = cnn_data['confidence']
            
            self.lbl_cnn_prediction.setText(f"Prediction: {pred_class}")
            self.lbl_cnn_confidence.setText(f"Confidence: {confidence:.2%}")
            
            # Update confidence bars
            for class_name, prob in cnn_data['probabilities'].items():
                self.confidence_bars[class_name].setValue(int(prob * 100))
            
            # Display GradCAM visualization
            if cnn_data.get('gradcam_overlay') is not None:
                self.display_numpy_image(cnn_data['gradcam_overlay'], self.lbl_gradcam_image)
        
        # Final decision
        final_text, final_color = self.get_final_decision(result)
        self.lbl_final_decision.setText(final_text)
        self.lbl_final_decision.setStyleSheet(f"""
            font-size: 16px; 
            font-weight: bold; 
            padding: 20px;
            border: 2px solid {final_color};
            border-radius: 5px;
            color: {final_color};
        """)
        
        # Processing time
        proc_time = result.get('processing_time', 0)
        self.lbl_processing_time.setText(f"Processing Time: {proc_time:.3f}s")
        
        self.log_message(f"Analysis complete - {final_text}")
        
        # Update defect detection visualization
        self.update_defect_detection_view(result)
        
        # Update LLM chat context
        if self.llm_chat:
            self.llm_chat.set_context(result)
        
    def display_tensor_image(self, tensor, label):
        """Convert tensor to QPixmap and display"""
        try:
            import numpy as np
            import cv2
            
            # Convert tensor to numpy
            if len(tensor.shape) == 3:
                img_array = tensor.permute(1, 2, 0).numpy()
            else:
                img_array = tensor.numpy()
                
            # Denormalize if needed
            img_array = (img_array * 255).astype(np.uint8)
            
            # Convert to QImage
            if len(img_array.shape) == 2:  # Grayscale
                height, width = img_array.shape
                qimage = QImage(img_array.data, width, height, width, QImage.Format_Grayscale8)
            else:  # RGB
                height, width, channels = img_array.shape
                bytes_per_line = channels * width
                qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.log_message(f"Error displaying image: {str(e)}")
    
    def display_numpy_image(self, img_array, label):
        """Convert numpy array to QPixmap and display"""
        try:
            import numpy as np
            
            # Ensure uint8
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            
            # Make sure array is contiguous
            img_array = np.ascontiguousarray(img_array)
            
            # Convert to QImage
            if len(img_array.shape) == 2:  # Grayscale
                height, width = img_array.shape
                qimage = QImage(img_array.data, width, height, width, QImage.Format_Grayscale8)
            else:  # RGB
                height, width, channels = img_array.shape
                bytes_per_line = channels * width
                qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.log_message(f"Error displaying numpy image: {str(e)}")
    
    def update_defect_detection_view(self, result):
        """Create combined visualization with GradCAM + Anomaly heatmap overlay and bounding box"""
        try:
            import numpy as np
            import cv2
            from PIL import Image, ImageDraw, ImageFont
            
            # Load original image
            img = Image.open(self.current_image_path).convert('RGB')
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
            
            # Apply Anomaly heatmap overlay if available (Autoencoder - Unsupervised)
            bounding_box = None
            if result['autoencoder'] and result['autoencoder'].get('heatmap') is not None:
                ae_heatmap = result['autoencoder']['heatmap']
                if hasattr(ae_heatmap, 'numpy'):
                    ae_heatmap = ae_heatmap.squeeze().numpy()
                
                # Resize to match image
                ae_heatmap = cv2.resize(ae_heatmap, (224, 224))
                
                # Normalize
                ae_heatmap = (ae_heatmap - ae_heatmap.min()) / (ae_heatmap.max() - ae_heatmap.min() + 1e-8)
                
                # Create colored heatmap (use different colormap to distinguish from GradCAM)
                ae_color = cv2.applyColorMap(np.uint8(255 * ae_heatmap), cv2.COLORMAP_HOT)
                ae_color = cv2.cvtColor(ae_color, cv2.COLOR_BGR2RGB) / 255.0
                combined = combined * 0.7 + ae_color * 0.3
                
                # Calculate bounding box from anomaly heatmap
                threshold = 0.5
                binary_mask = (ae_heatmap > threshold).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    bounding_box = (x, y, w, h)
            
            # Convert to uint8
            combined = np.clip(combined * 255, 0, 255).astype(np.uint8)
            
            # Resize for display
            combined = cv2.resize(combined, (600, 600))
            
            # Draw bounding box if detected
            if bounding_box:
                x, y, w, h = bounding_box
                # Scale bounding box to display size
                scale = 600 / 224
                x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
                cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(combined, "Anomaly Region", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Prepare detection results text for the summary label
            summary_lines = []
            
            # Add ground truth if available
            if self.current_image_path:
                img_path = Path(self.current_image_path)
                gt_class = img_path.parent.name if img_path.parent.name in ['CR', 'LP', 'ND', 'PO'] else None
                if gt_class:
                    gt_color = "#44ff44" if gt_class == 'ND' else "#ff9944"
                    summary_lines.append(f"<b style='color:{gt_color}'>Ground Truth: {gt_class}</b>")
            
            if result['autoencoder']:
                ae_data = result['autoencoder']
                status = "ANOMALY" if ae_data['is_anomaly'] else "NORMAL"
                summary_lines.append(f"<b>Autoencoder:</b> {status} (error: {ae_data['error']:.4f})")
            
            if result['cnn']:
                cnn_data = result['cnn']
                pred_class = cnn_data['prediction']
                # Check if prediction matches ground truth
                if self.current_image_path:
                    img_path = Path(self.current_image_path)
                    gt_class = img_path.parent.name if img_path.parent.name in ['CR', 'LP', 'ND', 'PO'] else None
                    if gt_class:
                        match_icon = "✓" if pred_class == gt_class else "✗"
                        match_color = "#44ff44" if pred_class == gt_class else "#ff4444"
                        summary_lines.append(f"<b>CNN:</b> {pred_class} ({cnn_data['confidence']:.1%}) <span style='color:{match_color}'>{match_icon}</span>")
                    else:
                        summary_lines.append(f"<b>CNN:</b> {pred_class} ({cnn_data['confidence']:.1%})")
                else:
                    summary_lines.append(f"<b>CNN:</b> {pred_class} ({cnn_data['confidence']:.1%})")
            
            final_text, final_color = self.get_final_decision(result)
            summary_lines.append(f"<b style='color:{final_color}'>FINAL: {final_text}</b>")
            
            # Update the results summary label
            self.lbl_results_summary.setText("<br>".join(summary_lines))
            
            # Display the combined image (without text overlay)
            self.display_numpy_image(combined, self.lbl_defect_detection)
            
        except Exception as e:
            self.log_message(f"Error creating combined view: {str(e)}")
            self.lbl_defect_detection.setText(f"Error: {str(e)}")
            
    def get_final_decision(self, result):
        """Determine final detection decision"""
        # If only CNN is used (SimpleCNN mode)
        if result['cnn'] and not result['autoencoder']:
            pred = result['cnn']['prediction']
            conf = result['cnn']['confidence']
            if pred == 'ND':  # No Defect
                return "NO DEFECT (NORMAL)", "#44ff44"
            else:
                return f"DEFECT DETECTED: {pred} ({conf:.1%})", "#ff4444"
        
        # If autoencoder detected an anomaly
        if result['autoencoder'] and result['autoencoder']['is_anomaly']:
            if result['cnn']:
                pred = result['cnn']['prediction']
                conf = result['cnn']['confidence']
                return f"DEFECT DETECTED: {pred} ({conf:.1%})", "#ff4444"
            else:
                return "ANOMALY DETECTED", "#ff4444"
        
        # Default: no anomaly detected
        return "NO DEFECT (NORMAL)", "#44ff44"
            
    def on_processing_finished(self):
        """Processing thread finished"""
        self.progress_bar.setVisible(False)
        self.btn_analyze.setEnabled(True)
        
    def export_results(self):
        """Export results to file"""
        if not self.current_result:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
            
        self.log_message("Export feature coming soon!")
        QMessageBox.information(self, "Info", "Export feature coming soon!")
        
    def clear_results(self):
        """Clear all results"""
        self.lbl_original_image.clear()
        self.lbl_original_image.setText("No Image Loaded")
        self.lbl_gradcam_image.clear()
        self.lbl_gradcam_image.setText("CNN GradCAM visualization\n(Supervised mode)")
        self.lbl_reconstructed_image.clear()
        self.lbl_reconstructed_image.setText("Autoencoder reconstruction\n(Unsupervised mode)")
        self.lbl_heatmap_image.clear()
        self.lbl_heatmap_image.setText("Anomaly heatmap\n(Unsupervised mode)")
        self.lbl_defect_detection.clear()
        self.lbl_defect_detection.setText("Combined detection results")
        self.lbl_results_summary.setText("Detection Results")
        
        self.lbl_ae_status.setText("Status: Not Analyzed")
        self.lbl_ae_error.setText("Reconstruction Error: -")
        self.lbl_cnn_prediction.setText("Prediction: Not Analyzed")
        self.lbl_cnn_confidence.setText("Confidence: -")
        self.lbl_final_decision.setText("AWAITING ANALYSIS")
        self.lbl_processing_time.setText("Processing Time: -")
        
        for bar in self.confidence_bars.values():
            bar.setValue(0)
            
        self.current_image_path = None
        self.current_result = None
        self.btn_analyze.setEnabled(False)
        self.log_message("Results cleared")
    
    def clear_all_results(self):
        """Clear all results including batch cache, statistics, and image list"""
        # Clear current results
        self.clear_results()
        
        # Clear batch cache
        self.batch_results_cache = {}
        self.batch_results = []
        
        # Clear image list and navigation
        self.image_list = []
        self.current_image_index = 0
        self.update_navigation_buttons()
        self.btn_analyze_all.setEnabled(False)
        
        # Reset statistics
        self.lbl_total_processed.setText("Processed: 0")
        self.lbl_defects_found.setText("Defects: 0")
        self.lbl_avg_time.setText("Avg Time: 0.00s")
        
        # Reset ground truth labels
        self.update_ground_truth_label(None)
        
        self.log_message("All results and cache cleared")
        self.status_bar.showMessage("All cleared - Ready for new analysis")
        
    def toggle_theme(self):
        """Toggle between dark and light theme"""
        self.log_message("Theme toggle coming soon!")
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About",
            "Weld Defect Detection System v1.0\n\n"
            "Professional AI-powered weld defect detection using:\n"
            "• Convolutional Autoencoder (Unsupervised)\n"
            "• CNN Classifier (Supervised)\n\n"
            "Detects: CR (Cracks), LP (Lack of Penetration), PO (Porosity)\n\n"
            "© 2026"
        )
        
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.txt_log.append(f"[{timestamp}] {message}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = WeldDefectGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
