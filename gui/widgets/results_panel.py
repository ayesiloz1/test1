"""
Results Panel Widget - Right side panel
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QProgressBar, QTextEdit, QTabWidget
)
from PyQt5.QtCore import Qt
from llm_chat import LLMChatWidget


class ResultsPanel(QWidget):
    """Right panel with detection results and AI assistant"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
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
        
        # Results tab
        tabs.addTab(self._create_results_tab(), "Results")
        
        # Metrics tab (for confusion matrices)
        self.metrics_tab = self._create_metrics_tab()
        tabs.addTab(self.metrics_tab, "Metrics")
        
        # AI Assistant tab
        self.llm_chat = LLMChatWidget()
        tabs.addTab(self.llm_chat, "AI Assistant")
        
        layout.addWidget(tabs)
    
    def _create_results_tab(self):
        """Create results tab content"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("DETECTION RESULTS")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Autoencoder section
        layout.addWidget(self._create_autoencoder_section())
        
        # CNN section
        layout.addWidget(self._create_cnn_section())
        
        # Final decision section
        layout.addWidget(self._create_decision_section())
        
        # Activity log section
        layout.addWidget(self._create_log_section())
        
        layout.addStretch()
        
        return panel
    
    def _create_autoencoder_section(self):
        """Create autoencoder results section"""
        group = QGroupBox("Autoencoder Analysis")
        layout = QVBoxLayout()
        
        self.lbl_ae_status = QLabel("Status: Not Analyzed")
        self.lbl_ae_status.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(self.lbl_ae_status)
        
        self.lbl_ae_error = QLabel("Reconstruction Error: -")
        layout.addWidget(self.lbl_ae_error)
        
        self.lbl_ae_threshold = QLabel("Threshold: -")
        layout.addWidget(self.lbl_ae_threshold)
        
        group.setLayout(layout)
        return group
    
    def _create_cnn_section(self):
        """Create CNN results section"""
        group = QGroupBox("CNN Classification")
        layout = QVBoxLayout()
        
        self.lbl_cnn_prediction = QLabel("Prediction: Not Analyzed")
        self.lbl_cnn_prediction.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(self.lbl_cnn_prediction)
        
        self.lbl_cnn_confidence = QLabel("Confidence: -")
        layout.addWidget(self.lbl_cnn_confidence)
        
        # Confidence bars
        conf_label = QLabel("Class Probabilities:")
        conf_label.setStyleSheet("margin-top: 10px;")
        layout.addWidget(conf_label)
        
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
            layout.addLayout(bar_layout)
            self.confidence_bars[class_name] = bar
        
        group.setLayout(layout)
        return group
    
    def _create_decision_section(self):
        """Create final decision section"""
        group = QGroupBox("Final Decision")
        layout = QVBoxLayout()
        
        self.lbl_final_decision = QLabel("AWAITING ANALYSIS")
        self.lbl_final_decision.setAlignment(Qt.AlignCenter)
        self.lbl_final_decision.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            padding: 20px;
            border: 2px solid #555;
            border-radius: 5px;
        """)
        layout.addWidget(self.lbl_final_decision)
        
        self.lbl_processing_time = QLabel("Processing Time: -")
        self.lbl_processing_time.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_processing_time)
        
        group.setLayout(layout)
        return group
    
    def _create_log_section(self):
        """Create activity log section"""
        group = QGroupBox("Activity Log")
        layout = QVBoxLayout()
        
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(200)
        layout.addWidget(self.txt_log)
        
        group.setLayout(layout)
        return group
    
    def log_message(self, message):
        """Add message to log"""
        self.txt_log.append(message)
    
    def update_ae_results(self, ae_data):
        """Update autoencoder results"""
        if ae_data:
            status = ae_data.get('status', 'UNKNOWN')
            error = ae_data.get('error', 0)
            threshold = ae_data.get('threshold', 0)
            
            # Set status with color
            if status == 'NORMAL':
                self.lbl_ae_status.setText(f"Status: {status}")
                self.lbl_ae_status.setStyleSheet("font-size: 13px; font-weight: bold; color: #44ff44;")
            else:
                self.lbl_ae_status.setText(f"Status: {status}")
                self.lbl_ae_status.setStyleSheet("font-size: 13px; font-weight: bold; color: #ff4444;")
            
            self.lbl_ae_error.setText(f"Reconstruction Error: {error:.6f}")
            self.lbl_ae_threshold.setText(f"Threshold: {threshold:.6f}")
    
    def update_cnn_results(self, cnn_data):
        """Update CNN results"""
        if cnn_data:
            prediction = cnn_data.get('prediction', 'Unknown')
            confidence = cnn_data.get('confidence', 0)
            probabilities = cnn_data.get('probabilities', {})
            
            self.lbl_cnn_prediction.setText(f"Prediction: {prediction}")
            self.lbl_cnn_confidence.setText(f"Confidence: {confidence:.1%}")
            
            # Update confidence bars
            for class_name, prob in probabilities.items():
                if class_name in self.confidence_bars:
                    self.confidence_bars[class_name].setValue(int(prob * 100))
    
    def update_final_decision(self, text, color):
        """Update final decision"""
        self.lbl_final_decision.setText(text)
        self.lbl_final_decision.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            padding: 20px;
            border: 2px solid {color};
            border-radius: 5px;
            color: {color};
        """)
    
    def update_processing_time(self, time_seconds):
        """Update processing time"""
        self.lbl_processing_time.setText(f"Processing Time: {time_seconds:.3f}s")
    
    def clear_results(self):
        """Clear all results"""
        self.lbl_ae_status.setText("Status: Not Analyzed")
        self.lbl_ae_status.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.lbl_ae_error.setText("Reconstruction Error: -")
        self.lbl_ae_threshold.setText("Threshold: -")
        
        self.lbl_cnn_prediction.setText("Prediction: Not Analyzed")
        self.lbl_cnn_confidence.setText("Confidence: -")
        
        for bar in self.confidence_bars.values():
            bar.setValue(0)
        
        self.lbl_final_decision.setText("AWAITING ANALYSIS")
        self.lbl_final_decision.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            padding: 20px;
            border: 2px solid #555;
            border-radius: 5px;
        """)
        self.lbl_processing_time.setText("Processing Time: -")
        
        # Clear metrics tab
        if hasattr(self, 'metrics_container'):
            for i in reversed(range(self.metrics_container.count())):
                widget = self.metrics_container.itemAt(i).widget()
                if widget:
                    widget.deleteLater()
    
    def _create_metrics_tab(self):
        """Create metrics tab with confusion matrices"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        title = QLabel("BATCH ANALYSIS METRICS")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Scrollable area for confusion matrices
        from PyQt5.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        metrics_widget = QWidget()
        self.metrics_container = QVBoxLayout(metrics_widget)
        
        # Placeholder
        placeholder = QLabel("Run 'Analyze All' to generate confusion matrices")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #888; padding: 50px; font-size: 12px;")
        self.metrics_container.addWidget(placeholder)
        self.metrics_container.addStretch()
        
        scroll.setWidget(metrics_widget)
        layout.addWidget(scroll)
        
        return panel
    
    def display_confusion_matrices(self, cnn_data, hybrid_data):
        """
        Display confusion matrices for CNN and Hybrid models
        
        Args:
            cnn_data: dict with 'y_true', 'y_pred', 'metrics'
            hybrid_data: dict with 'y_true', 'y_pred', 'metrics'
        """
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from utils.metrics import create_confusion_matrix_figure
        
        # Clear previous content
        for i in reversed(range(self.metrics_container.count())):
            widget = self.metrics_container.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        class_names = ['CR', 'LP', 'ND', 'PO']
        
        # CNN Confusion Matrix
        if cnn_data and len(cnn_data['y_true']) > 0:
            cnn_label = QLabel("CNN Only (Supervised)")
            cnn_label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 10px;")
            self.metrics_container.addWidget(cnn_label)
            
            cnn_fig = create_confusion_matrix_figure(
                cnn_data['y_true'], 
                cnn_data['y_pred'],
                class_names,
                "CNN Confusion Matrix"
            )
            cnn_canvas = FigureCanvasQTAgg(cnn_fig)
            cnn_canvas.setMinimumHeight(400)
            self.metrics_container.addWidget(cnn_canvas)
            
            # CNN Metrics
            metrics = cnn_data['metrics']
            cnn_metrics_label = QLabel(
                f"Accuracy: {metrics['accuracy']:.2%} | "
                f"Precision: {metrics['precision']:.2%} | "
                f"Recall: {metrics['recall']:.2%} | "
                f"F1: {metrics['f1_score']:.2%}"
            )
            cnn_metrics_label.setStyleSheet("font-size: 12px; padding: 5px; color: #aaa;")
            self.metrics_container.addWidget(cnn_metrics_label)
        
        # Hybrid Confusion Matrix
        if hybrid_data and len(hybrid_data['y_true']) > 0:
            hybrid_label = QLabel("Hybrid (CNN + Autoencoder)")
            hybrid_label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 10px; margin-top: 20px;")
            self.metrics_container.addWidget(hybrid_label)
            
            hybrid_fig = create_confusion_matrix_figure(
                hybrid_data['y_true'],
                hybrid_data['y_pred'],
                class_names,
                "Hybrid Confusion Matrix"
            )
            hybrid_canvas = FigureCanvasQTAgg(hybrid_fig)
            hybrid_canvas.setMinimumHeight(400)
            self.metrics_container.addWidget(hybrid_canvas)
            
            # Hybrid Metrics
            metrics = hybrid_data['metrics']
            hybrid_metrics_label = QLabel(
                f"Accuracy: {metrics['accuracy']:.2%} | "
                f"Precision: {metrics['precision']:.2%} | "
                f"Recall: {metrics['recall']:.2%} | "
                f"F1: {metrics['f1_score']:.2%}"
            )
            hybrid_metrics_label.setStyleSheet("font-size: 12px; padding: 5px; color: #aaa;")
            self.metrics_container.addWidget(hybrid_metrics_label)
        
        self.metrics_container.addStretch()
