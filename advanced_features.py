"""
Advanced Detection Features
Includes tracking, zone detection, and export capabilities
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import json
from datetime import datetime
import csv

class ObjectTracker:
    """Enhanced object tracking with history"""
    
    def __init__(self, max_history=30):
        self.tracks = defaultdict(lambda: deque(maxlen=max_history))
        self.next_id = 0
        self.max_distance = 50
        
    def update(self, detections):
        """Update tracks with new detections"""
        if not detections:
            return []
        
        # Simple centroid-based tracking
        current_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_centroids.append((cx, cy, det))
        
        # Match with existing tracks
        tracked_objects = []
        for cx, cy, det in current_centroids:
            best_match = None
            min_dist = self.max_distance
            
            for track_id, history in self.tracks.items():
                if history:
                    last_cx, last_cy = history[-1]
                    dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = track_id
            
            if best_match is None:
                best_match = self.next_id
                self.next_id += 1
            
            self.tracks[best_match].append((cx, cy))
            det['track_id'] = best_match
            tracked_objects.append(det)
        
        return tracked_objects
    
    def draw_tracks(self, frame, tracked_objects):
        """Draw tracking trails on frame"""
        for obj in tracked_objects:
            track_id = obj['track_id']
            if track_id in self.tracks:
                points = list(self.tracks[track_id])
                for i in range(1, len(points)):
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        return frame


class DetectionZone:
    """Define zones for specific detection areas"""
    
    def __init__(self, points, name="Zone 1"):
        self.points = np.array(points, dtype=np.int32)
        self.name = name
        self.color = (0, 255, 255)
        self.detections_count = 0
        
    def is_inside(self, point):
        """Check if point is inside zone"""
        return cv2.pointPolygonTest(self.points, point, False) >= 0
    
    def draw(self, frame, alpha=0.3):
        """Draw zone on frame"""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.points], self.color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [self.points], True, self.color, 2)
        
        # Draw zone name
        x, y = self.points[0]
        cv2.putText(frame, f"{self.name}: {self.detections_count}",
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, self.color, 2)
        
        return frame
    
    def count_detections(self, detections):
        """Count detections inside zone"""
        count = 0
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if self.is_inside((cx, cy)):
                count += 1
        
        self.detections_count = count
        return count


class DetectionExporter:
    """Export detection results to various formats"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self.detections_log = []
        
    def log_detection(self, frame_number, timestamp, detections):
        """Log detection for current frame"""
        entry = {
            'frame': frame_number,
            'timestamp': timestamp,
            'detections': detections
        }
        self.detections_log.append(entry)
    
    def export_json(self, filename=None):
        """Export to JSON format"""
        if filename is None:
            filename = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, 'w') as f:
            json.dump(self.detections_log, f, indent=2)
        
        return filepath
    
    def export_csv(self, filename=None):
        """Export to CSV format"""
        if filename is None:
            filename = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = f"{self.output_dir}/{filename}"
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp', 'Class', 'Confidence', 
                           'X1', 'Y1', 'X2', 'Y2', 'Track_ID'])
            
            for entry in self.detections_log:
                frame = entry['frame']
                timestamp = entry['timestamp']
                for det in entry['detections']:
                    x1, y1, x2, y2 = det['bbox']
                    writer.writerow([
                        frame, timestamp, det['class_name'],
                        det['confidence'], x1, y1, x2, y2,
                        det.get('track_id', 'N/A')
                    ])
        
        return filepath
    
    def export_summary(self, filename=None):
        """Export summary statistics"""
        if filename is None:
            filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath = f"{self.output_dir}/{filename}"
        
        # Calculate statistics
        total_frames = len(self.detections_log)
        class_counts = defaultdict(int)
        
        for entry in self.detections_log:
            for det in entry['detections']:
                class_counts[det['class_name']] += 1
        
        with open(filepath, 'w') as f:
            f.write("YOLO Detection Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Frames Processed: {total_frames}\n")
            f.write(f"Total Detections: {sum(class_counts.values())}\n\n")
            f.write("Detections by Class:\n")
            f.write("-" * 50 + "\n")
            
            for class_name, count in sorted(class_counts.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True):
                f.write(f"  {class_name}: {count}\n")
        
        return filepath


class HeatmapGenerator:
    """Generate detection heatmap"""
    
    def __init__(self, frame_shape):
        self.height, self.width = frame_shape[:2]
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        
    def update(self, detections):
        """Update heatmap with new detections"""
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            self.heatmap[y1:y2, x1:x2] += 1
    
    def get_heatmap(self, normalize=True):
        """Get heatmap visualization"""
        if normalize and self.heatmap.max() > 0:
            normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
        else:
            normalized = self.heatmap.astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def overlay_on_frame(self, frame, alpha=0.4):
        """Overlay heatmap on frame"""
        heatmap = self.get_heatmap()
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        return cv2.addWeighted(frame, 1 - alpha, heatmap_resized, alpha, 0)


class FPSStabilizer:
    """Stabilize FPS display"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.fps_values = deque(maxlen=window_size)
        
    def update(self, fps):
        """Add new FPS value"""
        self.fps_values.append(fps)
    
    def get_stable_fps(self):
        """Get moving average FPS"""
        if not self.fps_values:
            return 0.0
        return sum(self.fps_values) / len(self.fps_values)


class AlertSystem:
    """Alert system for specific detections"""
    
    def __init__(self):
        self.alert_classes = set()
        self.alert_threshold = 1
        self.alerts = []
        
    def add_alert_class(self, class_name, threshold=1):
        """Add class to alert on"""
        self.alert_classes.add(class_name)
        self.alert_threshold = threshold
    
    def check_alerts(self, detections):
        """Check if any alerts should be triggered"""
        class_counts = defaultdict(int)
        
        for det in detections:
            if det['class_name'] in self.alert_classes:
                class_counts[det['class_name']] += 1
        
        alerts = []
        for class_name, count in class_counts.items():
            if count >= self.alert_threshold:
                alert = {
                    'class': class_name,
                    'count': count,
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
                self.alerts.append(alert)
        
        return alerts
