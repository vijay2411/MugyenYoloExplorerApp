"""
RTSP Stream Configuration Dialog
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QGroupBox)

class RTSPDialog(QDialog):
    """Dialog for configuring RTSP stream"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RTSP Stream Configuration")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # RTSP URL Input
        url_group = QGroupBox("RTSP Stream URL")
        url_layout = QVBoxLayout()
        
        info_label = QLabel("Enter RTSP stream URL:")
        url_layout.addWidget(info_label)
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("rtsp://username:password@ip:port/stream")
        url_layout.addWidget(self.url_input)
        
        # Example URLs
        example_label = QLabel("Examples:")
        example_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        url_layout.addWidget(example_label)
        
        examples = [
            "rtsp://admin:password@192.168.1.64:554/stream1",
            "rtsp://192.168.1.100:8554/live",
            "rtsp://camera.local/h264"
        ]
        
        for example in examples:
            ex_label = QLabel(f"  • {example}")
            ex_label.setStyleSheet("color: #666; font-size: 9pt;")
            url_layout.addWidget(ex_label)
        
        url_group.setLayout(url_layout)
        layout.addWidget(url_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("Connect")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_url(self):
        """Get entered RTSP URL"""
        return self.url_input.text().strip()
