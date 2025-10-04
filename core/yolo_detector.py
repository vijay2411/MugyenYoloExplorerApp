"""
YOLO Detector Core Module
Handles YOLO model inference and detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import platform
import os

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', device='cpu'):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file
            device: Device to run inference on ('cpu', 'cuda', 'mps', 'npu', '0', etc.)
        """
        self.device = device
        print(f"🔧 Using device: {self.device}")
        
        # Load model with proper device handling
        try:
            print(f"🔄 Loading YOLO model on {self.device}...")
            self.model = YOLO(model_path)
            # Move model to device if not CPU
            if self.device != 'cpu':
                self.model.to(self.device)
            print(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load model on {self.device}, falling back to CPU")
            print(f"   Error: {str(e)}")
            self.device = 'cpu'
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully on {self.device}")
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # Statistics
        self.fps = 0
        self.inference_time = 0
        self.objects_detected = 0
        
        # Color map for different classes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    def get_available_devices(self):
        """Get list of available devices for inference"""
        devices = ['cpu']  # CPU is always available
        
        # Check CUDA availability
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            print(f"✅ CUDA available: {cuda_count} GPU(s)")
            for i in range(cuda_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                devices.append(f'cuda:{i}')
        
        # Check MPS (Mac Metal Performance Shaders) availability
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS (Mac GPU) available")
            devices.append('mps')
        
        # Check for NPU/other accelerators (Jetson Orin, etc.)
        if self._check_npu_availability():
            print("✅ NPU accelerator detected")
            devices.append('npu')
        
        return devices
    
    def _check_npu_availability(self):
        """Check for NPU availability (Jetson Orin, etc.)"""
        try:
            # Check for Jetson Orin NPU
            if os.path.exists('/sys/devices/platform/17000000.nvdla'):
                return True
            
            # Check for Jetson Xavier NPU
            if os.path.exists('/sys/devices/platform/17000000.nvdla'):
                return True
                
            # Check for other NPU indicators
            if 'npu' in platform.machine().lower():
                return True
                
            # Check environment variables for NPU
            if os.environ.get('NPU_AVAILABLE', '').lower() == 'true':
                return True
                
            # Check for Jetson-specific environment
            if os.environ.get('JETSON_NANO', '').lower() == 'true':
                return True
            if os.environ.get('JETSON_XAVIER', '').lower() == 'true':
                return True
            if os.environ.get('JETSON_ORIN', '').lower() == 'true':
                return True
                
        except Exception:
            pass
        
        return False
        
    def set_confidence(self, confidence):
        """Set confidence threshold"""
        self.confidence_threshold = confidence
    
    def set_iou(self, iou):
        """Set IOU threshold"""
        self.iou_threshold = iou
    
    def detect(self, frame):
        """
        Run detection on frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            results: Detection results
            stats: Statistics dictionary
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Calculate inference time
        self.inference_time = (time.time() - start_time) * 1000  # ms
        
        # Calculate FPS
        self.fps = 1000 / self.inference_time if self.inference_time > 0 else 0
        
        # Count objects
        self.objects_detected = len(results.boxes) if results.boxes is not None else 0
        
        # Prepare statistics
        stats = {
            'fps': self.fps,
            'inference_time': self.inference_time,
            'objects_detected': self.objects_detected,
            'detections': self._parse_detections(results)
        }
        
        return results, stats
    
    def _parse_detections(self, results):
        """Parse detection results into dictionary"""
        detections = {}
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                confidence = float(box.conf[0])
                
                if cls_name in detections:
                    detections[cls_name]['count'] += 1
                    detections[cls_name]['confidences'].append(confidence)
                else:
                    detections[cls_name] = {
                        'count': 1,
                        'confidences': [confidence]
                    }
        
        return detections
    
    def draw_results(self, frame, results):
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            results: Detection results from YOLO
            
        Returns:
            annotated_frame: Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                
                # Get class name and color
                class_name = self.model.names[cls_id]
                color = tuple(map(int, self.colors[cls_id % len(self.colors)]))
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                label = f"{class_name} {conf:.2f}"
                
                # Calculate label size
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        return annotated_frame
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_name': self.model.model_name,
            'device': self.device,
            'classes': len(self.model.names),
            'class_names': list(self.model.names.values())
        }
