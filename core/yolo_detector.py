"""
YOLO Detector Core Module
Optimized for Mac (MPS) and Jetson (TensorRT) devices
Handles YOLO model inference and detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import platform
import os
import subprocess

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', device='cpu'):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file
            device: Device to run inference on ('cpu', 'cuda', 'mps', 'npu', '0', etc.)
        """
        # Auto-detect best device if cpu is default but better options exist
        self.device = device
        print(f"🔧 Using device: {self.device}")
        
        # Check if running on Jetson
        self.is_jetson = self._detect_jetson()
        self.jetson_model = None
        self.model_path = model_path
        
        # Load model with proper device handling (auto TensorRT export on Jetson)
        self._load_model(model_path)
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # Statistics
        self.fps = 0
        self.inference_time = 0
        self.objects_detected = 0
        
        # Color map for different classes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    def _auto_select_device(self):
        """Automatically select the best available device"""
        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return '0'  # Use first CUDA device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _detect_jetson(self):
        """Detect if running on NVIDIA Jetson device"""
        try:
            # Check for Jetson-specific files
            jetson_paths = [
                '/etc/nv_tegra_release',  # Jetson release info
                '/sys/devices/soc0/family',  # SoC family
                '/proc/device-tree/model',  # Device model
            ]
            
            for path in jetson_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        content = f.read().lower()
                        if 'tegra' in content or 'jetson' in content or 'nvidia' in content:
                            print("🤖 Jetson device detected!")
                            self._print_jetson_info()
                            return True
            
            # Check environment variables
            jetson_env_vars = ['JETSON_NANO', 'JETSON_XAVIER', 'JETSON_ORIN', 'JETSON_MODEL']
            for var in jetson_env_vars:
                if os.environ.get(var):
                    print(f"🤖 Jetson device detected via {var}!")
                    return True
                    
        except Exception as e:
            print(f"⚠️ Error checking for Jetson: {e}")
        
        return False
    
    def _print_jetson_info(self):
        """Print Jetson device information"""
        try:
            # Try to get Jetson model
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().strip('\x00')
                    print(f"   Model: {model}")
            
            # Try to get L4T version (Linux for Tegra)
            if os.path.exists('/etc/nv_tegra_release'):
                with open('/etc/nv_tegra_release', 'r') as f:
                    release = f.read().strip()
                    print(f"   Release: {release}")
            
            # Check for DLA (Deep Learning Accelerator)
            dla_paths = [
                '/sys/devices/platform/13e10000.host1x/15880000.nvdla0',  # Xavier
                '/sys/devices/platform/13e00000.host1x/15880000.nvdla0',  # Orin
            ]
            for dla_path in dla_paths:
                if os.path.exists(dla_path):
                    print(f"   ✅ DLA accelerator found at {dla_path}")
                    
        except Exception as e:
            print(f"   ⚠️ Could not read Jetson info: {e}")
    
    def _load_model(self, model_path):
        """Load YOLO model with device-specific optimizations"""
        try:
            print(f"🔄 Loading YOLO model...")
            
            # For Jetson devices with CUDA, automatically use or create TensorRT engine
            if self.is_jetson and self.device in ['0', 'cuda', 'cuda:0']:
                trt_model_path = model_path.replace('.pt', '.engine')
                
                if os.path.exists(trt_model_path):
                    print(f"🚀 Loading existing TensorRT engine: {trt_model_path}")
                    try:
                        self.model = YOLO(trt_model_path, task='detect')
                        print(f"✅ TensorRT model loaded successfully!")
                        return
                    except Exception as e:
                        print(f"⚠️ Failed to load TensorRT engine: {e}")
                        print(f"   Falling back to PyTorch model...")
                
                # TensorRT engine doesn't exist - create it automatically
                if model_path.endswith('.pt'):
                    print(f"🔧 TensorRT engine not found. Creating one automatically...")
                    print(f"   This will take a few minutes but only happens once...")
                    
                    # Load PyTorch model first
                    temp_model = YOLO(model_path)
                    
                    try:
                        # Export to TensorRT
                        print(f"   Exporting to TensorRT (FP16 for better performance)...")
                        temp_model.export(
                            format='engine',
                            device=0,
                            half=True,  # FP16 for Jetson
                            workspace=4,
                            simplify=True
                        )
                        print(f"✅ TensorRT engine created: {trt_model_path}")
                        print(f"   Loading TensorRT engine...")
                        
                        # Load the newly created TensorRT engine
                        self.model = YOLO(trt_model_path, task='detect')
                        print(f"✅ TensorRT model loaded successfully!")
                        print(f"   💡 Future runs will be faster (engine already created)")
                        return
                        
                    except Exception as e:
                        print(f"⚠️ TensorRT export failed: {e}")
                        print(f"   Using standard PyTorch model (slower on Jetson)")
                        self.model = temp_model
                        self.model.to(self.device)
                        return
            
            # Load standard PyTorch model for non-Jetson or non-CUDA devices
            self.model = YOLO(model_path)
            
            # Move model to device
            if self.device == 'mps':
                # Mac MPS handling
                print(f"🍎 Optimizing for Mac Metal (MPS)...")
                try:
                    self.model.to(self.device)
                    print(f"✅ Model loaded successfully on MPS")
                except Exception as e:
                    print(f"⚠️ MPS optimization failed, using CPU: {e}")
                    self.device = 'cpu'
                    self.model.to(self.device)
                    
            elif self.device != 'cpu':
                # CUDA/GPU handling
                self.model.to(self.device)
                if torch.cuda.is_available() and (self.device.startswith('cuda') or self.device.isdigit()):
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"✅ Model loaded on GPU: {gpu_name}")
                else:
                    print(f"✅ Model loaded on device: {self.device}")
            else:
                # CPU
                print(f"✅ Model loaded on CPU")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load model on {self.device}")
            print(f"   Error: {str(e)}")
            print(f"   Falling back to CPU...")
            self.device = 'cpu'
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully on CPU")
    
    def get_available_devices(self):
        """Get list of available devices for inference"""
        devices = ['cpu']  # CPU is always available
        device_info = []
        
        # Check CUDA availability
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            print(f"\n✅ CUDA available: {cuda_count} GPU(s)")
            for i in range(cuda_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                devices.append(str(i))
                devices.append(f'cuda:{i}')
                device_info.append({
                    'id': str(i),
                    'name': gpu_name,
                    'memory_gb': gpu_memory,
                    'type': 'CUDA'
                })
        
        # Check MPS (Mac Metal Performance Shaders) availability
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("\n✅ MPS (Mac GPU) available")
            try:
                # Check if MPS is actually usable
                test_tensor = torch.zeros(1).to('mps')
                devices.append('mps')
                device_info.append({
                    'id': 'mps',
                    'name': 'Apple Metal Performance Shaders',
                    'type': 'MPS'
                })
                print("   Apple Silicon GPU acceleration enabled")
            except Exception as e:
                print(f"   ⚠️ MPS detected but not usable: {e}")
        
        # Jetson-specific info
        if self.is_jetson:
            print("\n🤖 Jetson Hardware Acceleration:")
            print("   • CUDA cores available for inference")
            print("   • Export to TensorRT for optimal performance")
            print("   • DLA engines may be available (model dependent)")
        
        print(f"\n📋 Available devices: {devices}\n")
        return devices, device_info
    
    def set_confidence(self, confidence):
        """Set confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, confidence))
    
    def set_iou(self, iou):
        """Set IOU threshold"""
        self.iou_threshold = max(0.0, min(1.0, iou))
    
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
        
        try:
            # Run inference - device is already set during model loading
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
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
                'detections': self._parse_detections(results),
                'device': self.device
            }
            
            return results, stats
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            # Return empty results
            return None, {
                'fps': 0,
                'inference_time': 0,
                'objects_detected': 0,
                'detections': {},
                'device': self.device,
                'error': str(e)
            }
    
    def _parse_detections(self, results):
        """Parse detection results into dictionary"""
        detections = {}
        
        if results and results.boxes is not None and len(results.boxes) > 0:
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
        if results is None:
            return frame.copy()
        
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
    
    def export_to_tensorrt(self, output_path=None, half=True, workspace=4):
        """
        Manually export model to TensorRT (usually done automatically on Jetson)
        
        Args:
            output_path: Path to save TensorRT engine (default: same as model with .engine)
            half: Use FP16 precision (recommended for Jetson)
            workspace: GPU memory workspace in GB
            
        Note: On Jetson devices with CUDA, TensorRT export happens automatically on first run.
              This method is for manual export or re-export with different settings.
        """
        if not self.is_jetson:
            print("⚠️ TensorRT export is optimized for Jetson devices")
            print("   You can still export, but may not see performance benefits")
        
        try:
            print(f"🔄 Manually exporting to TensorRT...")
            print(f"   Precision: {'FP16' if half else 'FP32'}")
            print(f"   Workspace: {workspace}GB")
            
            # Load PyTorch model if currently using TensorRT
            if hasattr(self, 'model_path') and self.model_path:
                temp_model = YOLO(self.model_path) if self.model_path.endswith('.pt') else self.model
            else:
                temp_model = self.model
            
            # Export model
            temp_model.export(
                format='engine',
                device=0,
                half=half,
                workspace=workspace,
                simplify=True
            )
            
            print(f"✅ TensorRT export complete!")
            print(f"   💡 Restart to use the new TensorRT engine")
            
        except Exception as e:
            print(f"❌ TensorRT export failed: {e}")
            print(f"   Make sure you have TensorRT installed on your device")
    
    def get_model_info(self):
        """Get model information"""
        info = {
            'model_name': str(self.model.model_name) if hasattr(self.model, 'model_name') else 'Unknown',
            'device': self.device,
            'is_jetson': self.is_jetson,
            'classes': len(self.model.names),
            'class_names': list(self.model.names.values()),
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold
        }
        
        # Add device-specific info
        if self.device == 'mps':
            info['device_type'] = 'Apple Silicon (MPS)'
        elif self.device.startswith('cuda') or self.device.isdigit():
            if torch.cuda.is_available():
                info['device_type'] = 'NVIDIA CUDA GPU'
                info['gpu_name'] = torch.cuda.get_device_name(0)
            else:
                info['device_type'] = 'GPU (unavailable)'
        else:
            info['device_type'] = 'CPU'
        
        return info
    
    def benchmark(self, frame, iterations=100):
        """
        Benchmark inference performance
        
        Args:
            frame: Test frame
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n🔬 Running benchmark with {iterations} iterations...")
        
        inference_times = []
        
        # Warmup
        for _ in range(10):
            self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Benchmark
        for i in range(iterations):
            start = time.time()
            self.model(frame, conf=self.confidence_threshold, verbose=False)
            inference_times.append((time.time() - start) * 1000)
            
            if (i + 1) % 25 == 0:
                print(f"   Progress: {i + 1}/{iterations}")
        
        results = {
            'avg_inference_time': np.mean(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'std_inference_time': np.std(inference_times),
            'avg_fps': 1000 / np.mean(inference_times),
            'device': self.device,
            'iterations': iterations
        }
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Device: {self.device}")
        print(f"   Avg Inference Time: {results['avg_inference_time']:.2f}ms")
        print(f"   Avg FPS: {results['avg_fps']:.2f}")
        print(f"   Min/Max Time: {results['min_inference_time']:.2f}ms / {results['max_inference_time']:.2f}ms")
        
        return results