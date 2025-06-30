import cv2
import torch
from ultralytics import YOLO
import os
import time
import json
import psutil

# Paths to models
DETECTION_MODEL_PATH = "yolo11n.pt"  # YOLO object detection model
CLASSIFICATION_MODEL_PATH = "best.pt"  # Your classifier model

# Classes mapping (update based on your classification model)
CLASS_NAMES = ["bus", "coupe", "crossover", "hatchback", "jeep", "mpv", "pickup-truck", "sedan", "suv", "taxi", "truck", "van", "vehicle", "wagon"]

# Initialize models
detection_model = YOLO(DETECTION_MODEL_PATH)
classification_model = YOLO(CLASSIFICATION_MODEL_PATH)

# Video input and output paths
VIDEO_SOURCE = "vdo.avi"
OUTPUT_FOLDER = "inference_output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Process video
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Video writer setup
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

    detection_results = detection_model(frame, classes=[2, 3, 5, 7])  # Car, motorcycle, bus, truck

    for result in detection_results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_img = frame[y1:y2, x1:x2]

            cls_result = classification_model(cropped_img, verbose=False)
            top_class_idx = cls_result[0].probs.top1
            class_name = CLASS_NAMES[top_class_idx]
            confidence_score = cls_result[0].probs.top1conf.item()

            predictions.append({
                "class_id": top_class_idx,
                "label": class_name,
                "confidence": round(confidence_score, 4),
                "bbox": [x1, y1, x2, y2]
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

# Final metrics
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

print("Inference complete and detailed metrics saved!")
