"""
Control Panel Widget - Left side controls
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QComboBox, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal


class ControlPanel(QWidget):
    """Left control panel with all buttons and controls"""
    
    # Signals
    load_image_clicked = pyqtSignal()
    load_folder_clicked = pyqtSignal()
    prev_clicked = pyqtSignal()
    next_clicked = pyqtSignal()
    class_filter_changed = pyqtSignal(str)
    review_mode_changed = pyqtSignal(str)
    detection_mode_changed = pyqtSignal(int)
    analyze_clicked = pyqtSignal()
    analyze_all_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    speed_changed = pyqtSignal(int)
    export_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    clear_all_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("CONTROL PANEL")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Input section
        layout.addWidget(self._create_input_section())
        
        # Detection mode section
        layout.addWidget(self._create_detection_mode_section())
        
        # Actions section
        layout.addWidget(self._create_actions_section())
        
        # Statistics section
        layout.addWidget(self._create_statistics_section())
        
        layout.addStretch()
    
    def _create_input_section(self):
        """Create input controls section"""
        group = QGroupBox("Input")
        layout = QVBoxLayout()
        
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image_clicked.emit)
        self.btn_load_image.setMinimumHeight(40)
        layout.addWidget(self.btn_load_image)
        
        self.btn_load_folder = QPushButton("Load Folder")
        self.btn_load_folder.clicked.connect(self.load_folder_clicked.emit)
        self.btn_load_folder.setMinimumHeight(40)
        layout.addWidget(self.btn_load_folder)
        
        # Navigation
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.clicked.connect(self.prev_clicked.emit)
        self.btn_prev.setEnabled(False)
        nav_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(self.next_clicked.emit)
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)
        
        # Class filter
        filter_label = QLabel("Filter by Class:")
        filter_label.setStyleSheet("font-size: 11px; color: #aaa; margin-top: 5px;")
        layout.addWidget(filter_label)
        
        self.combo_class_filter = QComboBox()
        self.combo_class_filter.addItems(["All", "CR", "LP", "ND", "PO"])
        self.combo_class_filter.currentTextChanged.connect(self.class_filter_changed.emit)
        self.combo_class_filter.setMinimumHeight(30)
        self.combo_class_filter.setEnabled(False)
        layout.addWidget(self.combo_class_filter)
        
        # Review mode filter
        review_label = QLabel("Review Mode:")
        review_label.setStyleSheet("font-size: 11px; color: #aaa; margin-top: 5px;")
        layout.addWidget(review_label)
        
        self.combo_review_mode = QComboBox()
        self.combo_review_mode.addItems([
            "All Images",
            "Misclassified Only",
            "Low Confidence (<70%)",
            "Issues (Misclassified or Low Conf.)"
        ])
        self.combo_review_mode.currentTextChanged.connect(self.review_mode_changed.emit)
        self.combo_review_mode.setMinimumHeight(30)
        self.combo_review_mode.setEnabled(False)
        layout.addWidget(self.combo_review_mode)
        
        # Counter
        self.lbl_image_counter = QLabel("No images loaded")
        self.lbl_image_counter.setAlignment(Qt.AlignCenter)
        self.lbl_image_counter.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.lbl_image_counter)
        
        group.setLayout(layout)
        return group
    
    def _create_detection_mode_section(self):
        """Create detection mode section"""
        group = QGroupBox("Detection Mode")
        layout = QVBoxLayout()
        
        self.combo_detection_mode = QComboBox()
        self.combo_detection_mode.addItems([
            "Supervised (CNN)",
            "Unsupervised (Autoencoder)",
            "Hybrid (Both)"
        ])
        self.combo_detection_mode.currentIndexChanged.connect(self.detection_mode_changed.emit)
        self.combo_detection_mode.setMinimumHeight(35)
        layout.addWidget(self.combo_detection_mode)
        
        self.lbl_mode_desc = QLabel("CNN classification with GradCAM visualization")
        self.lbl_mode_desc.setWordWrap(True)
        self.lbl_mode_desc.setStyleSheet("color: #888; font-size: 10px; padding: 5px;")
        layout.addWidget(self.lbl_mode_desc)
        
        group.setLayout(layout)
        return group
    
    def _create_actions_section(self):
        """Create actions buttons section"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()
        
        self.btn_analyze = QPushButton("Analyze Image")
        self.btn_analyze.clicked.connect(self.analyze_clicked.emit)
        self.btn_analyze.setMinimumHeight(50)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.btn_analyze)
        
        self.btn_analyze_all = QPushButton("Analyze All")
        self.btn_analyze_all.clicked.connect(self.analyze_all_clicked.emit)
        self.btn_analyze_all.setMinimumHeight(45)
        self.btn_analyze_all.setEnabled(False)
        self.btn_analyze_all.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.btn_analyze_all)
        
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause_clicked.emit)
        self.btn_pause.setMinimumHeight(40)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet("font-size: 12px; background-color: #cc7700;")
        layout.addWidget(self.btn_pause)
        
        # Processing speed control
        speed_label = QLabel("Processing Speed (delay):")
        speed_label.setStyleSheet("font-size: 11px; margin-top: 5px;")
        layout.addWidget(speed_label)
        
        speed_layout = QHBoxLayout()
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setMinimum(0)
        self.slider_speed.setMaximum(50)
        self.slider_speed.setValue(0)
        self.slider_speed.setTickPosition(QSlider.TicksBelow)
        self.slider_speed.setTickInterval(10)
        self.slider_speed.valueChanged.connect(self.speed_changed.emit)
        speed_layout.addWidget(self.slider_speed)
        
        self.lbl_speed_value = QLabel("0.0s")
        self.lbl_speed_value.setStyleSheet("font-size: 11px; min-width: 35px;")
        self.lbl_speed_value.setAlignment(Qt.AlignRight)
        speed_layout.addWidget(self.lbl_speed_value)
        layout.addLayout(speed_layout)
        
        self.btn_export = QPushButton("Export Results")
        self.btn_export.clicked.connect(self.export_clicked.emit)
        self.btn_export.setMinimumHeight(40)
        layout.addWidget(self.btn_export)
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_clicked.emit)
        self.btn_clear.setMinimumHeight(40)
        layout.addWidget(self.btn_clear)
        
        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.clear_all_clicked.emit)
        self.btn_clear_all.setMinimumHeight(40)
        self.btn_clear_all.setStyleSheet("color: #ff6666;")
        layout.addWidget(self.btn_clear_all)
        
        group.setLayout(layout)
        return group
    
    def _create_statistics_section(self):
        """Create statistics section"""
        group = QGroupBox("Statistics")
        layout = QVBoxLayout()
        
        self.lbl_total_processed = QLabel("Processed: 0")
        layout.addWidget(self.lbl_total_processed)
        
        self.lbl_defects_found = QLabel("Defects: 0")
        layout.addWidget(self.lbl_defects_found)
        
        self.lbl_avg_time = QLabel("Avg Time: 0.00s")
        layout.addWidget(self.lbl_avg_time)
        
        group.setLayout(layout)
        return group
