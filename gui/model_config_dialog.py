"""
Model Configuration Dialog
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QPushButton, QGroupBox, QRadioButton,
                             QButtonGroup)
from PyQt6.QtCore import Qt

class ModelConfigDialog(QDialog):
    """Dialog for configuring YOLO model"""
    
    def __init__(self, parent=None, custom_model_path=None):
        super().__init__(parent)
        self.setWindowTitle("Configure YOLO Model")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.custom_model_path = custom_model_path
        self.model_info = {}
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Model Selection
        model_group = QGroupBox("Select Model")
        model_layout = QVBoxLayout()
        
        if self.custom_model_path:
            # Custom model mode
            model_label = QLabel("Custom Model:")
            model_layout.addWidget(model_label)
            
            custom_model_label = QLabel(f"File: {self.custom_model_path.split('/')[-1]}")
            custom_model_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            model_layout.addWidget(custom_model_label)
            
            # Hide the combo box for custom models
            self.model_combo = None
        else:
            # Standard model mode
            model_label = QLabel("YOLO Model:")
            model_layout.addWidget(model_label)
            
            self.model_combo = QComboBox()
            self.model_combo.addItems([
                "YOLOv8 Nano (yolov8n.pt)",
                "YOLOv8 Small (yolov8s.pt)",
                "YOLOv8 Medium (yolov8m.pt)",
                "YOLOv8 Large (yolov8l.pt)",
                "YOLOv8 XLarge (yolov8x.pt)",
                "YOLOv11 Nano (yolo11n.pt)",
                "YOLOv11 Small (yolo11s.pt)",
                "YOLOv11 Medium (yolo11m.pt)",
                "YOLOv11 Large (yolo11l.pt)",
                "YOLOv11 XLarge (yolo11x.pt)"
            ])
            model_layout.addWidget(self.model_combo)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Device Selection
        device_group = QGroupBox("Select Device")
        device_layout = QVBoxLayout()
        
        self.device_buttons = QButtonGroup()
        
        # CPU option (default)
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setChecked(True)
        self.device_buttons.addButton(self.cpu_radio, 0)
        device_layout.addWidget(self.cpu_radio)
        
        # GPU options
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        self.device_buttons.addButton(self.gpu_radio, 1)
        device_layout.addWidget(self.gpu_radio)
        
        self.mps_radio = QRadioButton("Mac GPU (MPS)")
        self.device_buttons.addButton(self.mps_radio, 2)
        device_layout.addWidget(self.mps_radio)
        
        self.npu_radio = QRadioButton("NPU (Jetson)")
        self.device_buttons.addButton(self.npu_radio, 3)
        device_layout.addWidget(self.npu_radio)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_model_info(self):
        """Get selected model information"""
        if self.custom_model_path:
            # Custom model mode
            model_name = f"Custom: {self.custom_model_path.split('/')[-1]}"
            model_path = self.custom_model_path
        else:
            # Standard model mode
            model_text = self.model_combo.currentText()
            model_name = model_text
            model_path = model_text.split('(')[1].split(')')[0]
        
        # Determine device based on selection
        if self.cpu_radio.isChecked():
            device = 'cpu'
        elif self.gpu_radio.isChecked():
            device = 'cuda'
        elif self.mps_radio.isChecked():
            device = 'mps'
        elif self.npu_radio.isChecked():
            device = 'npu'
        else:
            device = 'cpu'  # Default fallback
        
        return {
            'name': model_name,
            'path': model_path,
            'device': device
        }