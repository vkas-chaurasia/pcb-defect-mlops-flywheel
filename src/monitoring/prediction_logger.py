import csv
import fcntl
import os
from datetime import datetime
from pathlib import Path

class PredictionLogger:
    def __init__(self, log_path: str = "monitoring/prediction_log.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.headers = [
            "timestamp",
            "num_detections",
            "avg_confidence",
            "max_confidence",
            "pred_open",
            "pred_short",
            "pred_mousebite",
            "pred_spur",
            "pred_spurious_copper",
            "pred_pin_hole",
            "avg_bbox_area",
            "pass_fail"
        ]
        
        # Initialize CSV with headers if it doesn't exist
        if not self.log_path.exists():
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, response) -> None:
        """Extracts proxy metrics from a PredictionResponse and logs to CSV."""
        
        detections = response.detections
        num_detections = len(detections)
        
        # Calculate confidences
        confidences = [d.confidence for d in detections]
        avg_conf = sum(confidences) / num_detections if num_detections > 0 else 0.0
        max_conf = max(confidences) if num_detections > 0 else 0.0
        
        # Count classes
        class_counts = {
            "open": 0, "short": 0, "mousebite": 0,
            "spur": 0, "spurious_copper": 0, "pin_hole": 0
        }
        for d in detections:
            if d.class_name in class_counts:
                class_counts[d.class_name] += 1
                
        # Calculate average bounding box area (normalized)
        areas = [d.bbox_xywhn[2] * d.bbox_xywhn[3] for d in detections]
        avg_area = sum(areas) / num_detections if num_detections > 0 else 0.0
        
        # Pass/Fail encoding (1 for FAIL/Defective, 0 for PASS)
        pass_fail_val = 1 if response.pass_fail == "FAIL" else 0
        
        row = [
            datetime.utcnow().isoformat(),
            num_detections,
            round(avg_conf, 4),
            round(max_conf, 4),
            class_counts["open"],
            class_counts["short"],
            class_counts["mousebite"],
            class_counts["spur"],
            class_counts["spurious_copper"],
            class_counts["pin_hole"],
            round(avg_area, 6),
            pass_fail_val
        ]
        
        # Thread-safe file append
        with open(self.log_path, mode="a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                writer = csv.writer(f)
                writer.writerow(row)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
