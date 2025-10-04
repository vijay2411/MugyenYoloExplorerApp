"""
Main Window for YOLO Explorer
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QComboBox, QSlider, QGroupBox,
                             QFileDialog, QMessageBox, QProgressBar, QApplication)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from core.yolo_detector import YOLODetector
from core.video_processor import VideoProcessor
from gui.widgets import StatsWidget, ControlPanel, SystemMonitorWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Explorer - Object Detection")
        
        # Get screen dimensions and set responsive window size
        self.setup_responsive_window()
        
        # Set application style with modern, readable color palette
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #ffffff;
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: #3c3c3c;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #64b5f6;
                font-weight: bold;
            }
            
            QLabel {
                color: #ffffff;
                font-size: 13px;
            }
            
            QPushButton {
                background-color: #2196f3;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            
            QComboBox {
                background-color: #4a4a4a;
                border: 2px solid #666666;
                border-radius: 6px;
                padding: 8px;
                color: #ffffff;
                font-size: 13px;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #2196f3;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #4a4a4a;
                border: 1px solid #666666;
                color: #ffffff;
                selection-background-color: #2196f3;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #666666;
                height: 8px;
                background: #4a4a4a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196f3;
                border: 2px solid #ffffff;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1976d2;
            }
            
            QProgressBar {
                border: 2px solid #4a4a4a;
                border-radius: 6px;
                text-align: center;
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 4px;
            }
            
            QTableWidget {
                background-color: #3c3c3c;
                alternate-background-color: #4a4a4a;
                border: 1px solid #666666;
                color: #ffffff;
                gridline-color: #555555;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #555555;
            }
            QTableWidget::item:selected {
                background-color: #2196f3;
            }
            QHeaderView::section {
                background-color: #4a4a4a;
                color: #ffffff;
                padding: 8px;
                border: 1px solid #666666;
                font-weight: bold;
            }
        """)
        
        # Initialize components
        self.detector = None
        self.video_processor = None
        self.current_frame = None
        self.is_running = False
        self.is_detecting = False
        
        self.init_ui()
        
        # Preload default YOLO model
        self.load_default_model()
    
    def setup_responsive_window(self):
        """Setup responsive window sizing based on screen dimensions"""
        # Get the primary screen
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        
        # Calculate responsive dimensions
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # Set window size as percentage of screen (80% width, 85% height)
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.85)
        
        # Ensure minimum size
        window_width = max(window_width, 1200)
        window_height = max(window_height, 800)
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.setGeometry(x, y, window_width, window_height)
        
        # Set minimum size to prevent window from being too small
        self.setMinimumSize(1000, 600)
        
        # Enable window resizing
        self.setWindowFlags(Qt.WindowType.Window)
        
        # Connect to screen change events for multi-monitor support
        QApplication.instance().primaryScreenChanged.connect(self.handle_screen_change)
        
        print(f"🖥️ Screen: {screen_width}x{screen_height}")
        print(f"📱 Window: {window_width}x{window_height}")
    
    def handle_screen_change(self):
        """Handle screen changes (e.g., connecting external monitors)"""
        print("🔄 Screen configuration changed, adjusting window...")
        self.setup_responsive_window()
    
    def resizeEvent(self, event):
        """Handle window resize events for responsive layout"""
        super().resizeEvent(event)
        
        # Update left panel width based on new window size
        if hasattr(self, 'left_panel'):
            new_width = int(self.width() * 0.35)
            self.left_panel.setMaximumWidth(new_width)
            print(f"🔄 Resized: Window {self.width()}x{self.height()}, Left panel max width: {new_width}")
        
        # Update video label minimum size
        if hasattr(self, 'video_label'):
            min_width = max(400, int(self.width() * 0.4))
            min_height = max(300, int(self.height() * 0.6))
            self.video_label.setMinimumSize(min_width, min_height)
            print(f"📹 Video label: {min_width}x{min_height}")
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left Panel - Controls (responsive width)
        self.left_panel = self.create_left_panel()
        # Set maximum width as percentage of window width
        self.left_panel.setMaximumWidth(int(self.width() * 0.35))
        main_layout.addWidget(self.left_panel, 1)
        
        # Right Panel - Video Display (responsive)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)
        
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Task Selection
        task_group = QGroupBox("Select Task")
        task_layout = QVBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Detection", "Segmentation", "Pose", "Tracking"])
        task_layout.addWidget(self.task_combo)
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)
        
        # Model Selection
        model_group = QGroupBox("Current Model")
        model_layout = QVBoxLayout()
        
        self.model_label = QLabel("No model loaded")
        self.model_label.setWordWrap(True)
        model_layout.addWidget(self.model_label)
        
        self.config_btn = QPushButton("Configure Model...")
        self.config_btn.clicked.connect(self.configure_model)
        model_layout.addWidget(self.config_btn)
        
        self.load_model_btn = QPushButton("Load Custom Model...")
        self.load_model_btn.clicked.connect(self.load_custom_model)
        model_layout.addWidget(self.load_model_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Input Source Selection
        source_group = QGroupBox("Select Input Source")
        source_layout = QVBoxLayout()
        
        self.source_combo = QComboBox()
        # Populate with available webcams first
        self.populate_webcam_sources()
        # Add other source types
        self.source_combo.addItems(["Video File", "Image", "RTSP Stream"])
        source_layout.addWidget(self.source_combo)
        
        # Button layout for source selection
        button_layout = QHBoxLayout()
        
        self.select_source_btn = QPushButton("Select Source")
        self.select_source_btn.clicked.connect(self.select_source)
        button_layout.addWidget(self.select_source_btn)
        
        self.refresh_cameras_btn = QPushButton("Refresh Cameras")
        self.refresh_cameras_btn.clicked.connect(self.refresh_cameras)
        self.refresh_cameras_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        button_layout.addWidget(self.refresh_cameras_btn)
        
        source_layout.addLayout(button_layout)
        
        self.source_info_label = QLabel("No source selected")
        self.source_info_label.setWordWrap(True)
        source_layout.addWidget(self.source_info_label)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Detection Controls
        detection_group = QGroupBox("Detection")
        detection_layout = QVBoxLayout()
        
        # Confidence Slider
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        conf_layout.addWidget(self.conf_slider)
        self.conf_value_label = QLabel("0.50")
        conf_layout.addWidget(self.conf_value_label)
        detection_layout.addLayout(conf_layout)
        
        # Start/Stop Buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start YOLO Detection")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        button_layout.addWidget(self.stop_btn)
        detection_layout.addLayout(button_layout)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Statistics
        self.stats_widget = StatsWidget()
        layout.addWidget(self.stats_widget)
        
        # System Monitor
        self.system_monitor = SystemMonitorWidget()
        layout.addWidget(self.system_monitor)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create right video display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Video Feed Label (responsive)
        self.video_label = QLabel()
        # Set minimum size as percentage of window
        min_width = max(400, int(self.width() * 0.4))
        min_height = max(300, int(self.height() * 0.6))
        self.video_label.setMinimumSize(min_width, min_height)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000; 
                border: 2px solid #333;
                border-radius: 5px;
            }
        """)
        self.video_label.setText("No Video Feed")
        layout.addWidget(self.video_label)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return panel
    
    def configure_model(self):
        """Configure YOLO model"""
        from gui.model_config_dialog import ModelConfigDialog
        dialog = ModelConfigDialog(self)
        if dialog.exec():
            model_info = dialog.get_model_info()
            self.load_yolo_model(model_info)
    
    def load_custom_model(self):
        """Load custom YOLO model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "",
            "Model Files (*.pt *.onnx);;All Files (*)"
        )
        
        if file_path:
            try:
                self.detector = YOLODetector(model_path=file_path)
                self.model_label.setText(f"Custom Model: {file_path.split('/')[-1]}")
                QMessageBox.information(self, "Success", "Model loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def load_yolo_model(self, model_info):
        """Load YOLO model with configuration"""
        try:
            self.detector = YOLODetector(
                model_path=model_info['path'],
                device=model_info.get('device', 'cpu')
            )
            self.model_label.setText(f"Model: {model_info['name']}\nDevice: {model_info.get('device', 'CPU')}")
            QMessageBox.information(self, "Success", "Model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def load_default_model(self):
        """Load default YOLO model on startup"""
        try:
            print("🔄 Loading default YOLO model...")
            self.detector = YOLODetector(model_path='yolov8n.pt', device='cpu')
            self.model_label.setText("Model: YOLOv8n (Default)\nDevice: CPU")
            print("✅ Default model loaded successfully")
        except Exception as e:
            print(f"❌ Warning: Could not load default model: {str(e)}")
            self.model_label.setText("Model: None loaded")
    
    def populate_webcam_sources(self):
        """Populate combo box with available webcam sources"""
        available_cameras = self.detect_webcams()
        if available_cameras:
            for camera_index in available_cameras:
                self.source_combo.addItem(f"Camera {camera_index}")
        else:
            self.source_combo.addItem("No cameras detected")
    
    def refresh_cameras(self):
        """Refresh the list of available cameras"""
        # Clear existing camera items (keep Video File, Image, RTSP Stream)
        while self.source_combo.count() > 3:
            self.source_combo.removeItem(0)
        
        # Re-populate with detected cameras
        available_cameras = self.detect_webcams()
        if available_cameras:
            for camera_index in available_cameras:
                self.source_combo.insertItem(0, f"Camera {camera_index}")
        else:
            self.source_combo.insertItem(0, "No cameras detected")
        
        # Show feedback
        if available_cameras:
            QMessageBox.information(self, "Camera Detection", f"Found {len(available_cameras)} camera(s): {available_cameras}")
        else:
            QMessageBox.warning(self, "Camera Detection", "No cameras detected. Please check your camera connections.")
    
    def select_source(self):
        """Select input source"""
        source_type = self.source_combo.currentText()
        
        # Check if it's a camera source (starts with "Camera")
        if source_type.startswith("Camera"):
            try:
                # Extract camera index from "Camera X" format
                camera_index = int(source_type.split()[1])
                self.video_processor = VideoProcessor(source=camera_index)
                self.source_info_label.setText(f"Source: {source_type}")
                # Auto-start video feed
                self.start_video_feed()
            except (ValueError, IndexError) as e:
                QMessageBox.warning(self, "Warning", f"Invalid camera selection: {str(e)}")
                return
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Camera {camera_index} failed to open: {str(e)}")
                return
        
        elif source_type == "Video File":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
            )
            if file_path:
                self.video_processor = VideoProcessor(source=file_path)
                self.source_info_label.setText(f"Source: {file_path.split('/')[-1]}")
                # Auto-start video feed
                self.start_video_feed()
        
        elif source_type == "Image":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image File", "",
                "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
            )
            if file_path:
                self.video_processor = VideoProcessor(source=file_path, is_image=True)
                self.source_info_label.setText(f"Source: {file_path.split('/')[-1]}")
                # Auto-start video feed
                self.start_video_feed()
        
        elif source_type == "RTSP Stream":
            from gui.rtsp_dialog import RTSPDialog
            dialog = RTSPDialog(self)
            if dialog.exec():
                rtsp_url = dialog.get_url()
                self.video_processor = VideoProcessor(source=rtsp_url)
                self.source_info_label.setText(f"Source: RTSP Stream")
                # Auto-start video feed
                self.start_video_feed()
    
    def update_confidence(self, value):
        """Update confidence threshold"""
        conf_value = value / 100.0
        self.conf_value_label.setText(f"{conf_value:.2f}")
        if self.detector:
            self.detector.set_confidence(conf_value)
    
    def start_video_feed(self):
        """Start video feed display (without detection)"""
        if not self.video_processor:
            return
            
        self.is_running = True
        self.is_detecting = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        # Start processing timer for video display only
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)  # ~33 FPS
        
        # Start system monitoring timer
        self.system_timer = QTimer()
        self.system_timer.timeout.connect(self.update_system_stats)
        self.system_timer.start(1000)  # Update every second
    
    def start_detection(self):
        """Start object detection"""
        if not self.video_processor:
            QMessageBox.warning(self, "Warning", "Please select an input source!")
            return
        
        # Enable YOLO detection
        self.is_detecting = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # If not already running, start the timer
        if not self.is_running:
            self.is_running = True
            self.timer = QTimer()
            self.timer.timeout.connect(self.process_frame)
            self.timer.start(30)  # ~33 FPS
    
    def stop_detection(self):
        """Stop object detection"""
        self.is_detecting = False
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        # Stop system monitoring timer
        if hasattr(self, 'system_timer') and self.system_timer.isActive():
            self.system_timer.stop()
        
        if self.video_processor:
            self.video_processor.release()
    
    def process_frame(self):
        """Process single frame"""
        if not self.is_running:
            return
        
        frame = self.video_processor.read_frame()
        
        if frame is None:
            self.stop_detection()
            return
        
        # Display frame first (always show video feed)
        self.display_frame(frame)
        
        # Run YOLO detection only if detection is enabled and detector is available
        if self.is_detecting and self.detector:
            results, stats = self.detector.detect(frame)
            
            # Draw results
            annotated_frame = self.detector.draw_results(frame, results)
            
            # Update display with annotated frame
            self.display_frame(annotated_frame)
            
            # Update statistics
            self.stats_widget.update_stats(stats)
            
            # Update system monitor with detection stats
            if hasattr(self, 'system_monitor'):
                self.system_monitor.update_detection_stats(stats)
    
    def update_system_stats(self):
        """Update system monitoring statistics"""
        if hasattr(self, 'system_monitor'):
            self.system_monitor.update_stats()
    
    def display_frame(self, frame):
        """Display frame in video label"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label while maintaining aspect ratio
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Center the pixmap in the label
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_detection()
        if self.video_processor:
            self.video_processor.release()
        event.accept()
    
    def detect_webcams(self):
        """Detect available webcam devices"""
        available_cameras = []
        print("🔍 Detecting available cameras...")
        
        # Check up to 10 camera indices (0-9)
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        print(f"✅ Camera {i}: {width}x{height} @ {fps:.1f} FPS")
                        available_cameras.append(i)
                    cap.release()
                else:
                    cap.release()
            except Exception as e:
                # Silently continue for non-existent cameras
                continue
        
        if available_cameras:
            print(f"📹 Found {len(available_cameras)} camera(s): {available_cameras}")
        else:
            print("❌ No cameras detected")
        
        return available_cameras
