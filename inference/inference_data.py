#"""
import cv2
import time
import json
import os
import psutil
from ultralytics import YOLO
import random

# === Config ===
MODEL_PATH = "best.pt"
VIDEO_SOURCE = "vdo.avi"
OUTPUT_FOLDER = "inference_output"

CONFIDENCE_THRESHOLD = 0.4
CLASS_NAMES = ["bus", "coupe", "crossover", "hatchback", "jeep", "mpv", "pickup-truck", "sedan", "suv", "taxi", "truck", "van", "vehicle", "wagon"]

# Define preferred colors for your classes (BGR format for OpenCV)
CLASS_COLOR_MAP = {
    "suv": (0, 0, 255),           # Red
    "mpv": (0, 165, 255),         # Orange
    "sedan": (0, 255, 255),       # Yellow
    "pickup-truck": (0, 255, 0),  # Green
    "truck": (255, 255, 0),       # Cyan-ish Yellow
    "bus": (255, 0, 0),           # Blue
    "van": (255, 0, 255),         # Magenta
    "taxi": (147, 20, 255),       # Violet
    "coupe": (255, 128, 0),       # Amber
    "wagon": (0, 128, 255),       # Sky Blue
    "vehicle": (128, 0, 255),     # Purple
    "crossover": (0, 255, 128),   # Mint Green
    "hatchback": (255, 105, 180), # Pink
    "jeep": (0, 128, 128)         # Teal
}





os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Load model ===
model = YOLO(MODEL_PATH)

# === Open video ===
cap = cv2.VideoCapture(VIDEO_SOURCE)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_fps = cap.get(cv2.CAP_PROP_FPS)
out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_video = cv2.VideoWriter(os.path.join(OUTPUT_FOLDER, "output_video.mp4"), fourcc, out_fps, (out_width, out_height))

log = []
frame_count = 0
start_time = time.time()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_data = {"frame_id": frame_count}
    predictions = []

    frame_start_time = time.perf_counter()
    results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        predictions.append({
            "class_id": cls_id,
            "label": label,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2]
        })

        color = CLASS_COLOR_MAP.get(label, (0, 255, 0))  # default: green if not found
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    frame_end_time = time.perf_counter()

    frame_data["predictions"] = predictions
    frame_data["inference_time_ms"] = round((frame_end_time - frame_start_time) * 1000, 2)
    frame_data["cpu_percent"] = psutil.cpu_percent()
    frame_data["ram_usage_mb"] = round(psutil.virtual_memory().used / (1024 * 1024), 2)

    log.append(frame_data)
    out_video.write(frame)
    frame_count += 1

cap.release()
out_video.release()

# === Save metrics ===
total_time = time.time() - start_time
fps = round(frame_count / total_time, 2)

output = {
    "video_path": VIDEO_SOURCE,
    "total_frames": frame_count,
    "average_fps": fps,
    "frames": log
}

with open(os.path.join(OUTPUT_FOLDER, "inference_metrics.json"), "w") as f:
    json.dump(output, f, indent=2)

print(f"Inference complete! Output saved to {OUTPUT_FOLDER}")

#"""

