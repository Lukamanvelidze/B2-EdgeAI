import cv2
import time
import json
import psutil
from ultralytics import YOLO

# Stage 1: Base YOLO model for bounding boxes
base_model = YOLO('yolo11n.pt')

# Stage 2: Custom YOLO model for classification
classifier = YOLO('global_model.pt')
classifier.model.names = {
    0: "bike", 1: "bus", 2: "caravan", 3: "coupe", 4: "crossover", 5: "hatchback",
    6: "jeep", 7: "mpv", 8: "pickup-truck", 9: "sedan", 10: "suv", 11: "taxi",
    12: "truck", 13: "van", 14: "vehicle", 15: "wagon"
}
label_map = classifier.model.names

# Define allowed transport class names from base model
transport_classes = {"car", "bicycle", "bus", "truck", "motorcycle"}
target_class_ids = {
    class_id for class_id, name in base_model.model.names.items()
    if name.lower() in transport_classes
}

video_path = '/home/luka/Desktop/AIC_2023_Track2/AIC23_Track2_NL_Retrieval/data/validation/S02/c007/vdo.avi'
delay = 1
log = []

frame_count = 0
start_time = time.time()
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_data = {"frame_id": frame_count}
    frame_start = time.perf_counter()
    predictions = []

    try:
        # Stage 1: detect bounding boxes
        base_results = base_model.predict(source=frame, imgsz=416, conf=0.25, stream=False)[0]

        for box, cls_id in zip(base_results.boxes.xyxy, base_results.boxes.cls):
            if int(cls_id.item()) not in target_class_ids:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            cropped = frame[y1:y2, x1:x2]

            # Stage 2: classify cropped object
            class_result = classifier.predict(source=cropped, imgsz=224, conf=0.3, stream=False, verbose=False)[0]
            if len(class_result.boxes) > 0:
                best_box = class_result.boxes[0]
                pred_cls = int(best_box.cls.item())
                conf = float(best_box.conf.item())
                label = label_map[pred_cls]
            else:
                pred_cls = -1
                label = "unknown"
                conf = 0.0

            predictions.append({
                "class_id": pred_cls,
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

    except Exception as e:
        print(f"Frame {frame_count} classification error: {e}")

    frame_end = time.perf_counter()
    frame_data["predictions"] = predictions
    frame_data["inference_time_ms"] = round((frame_end - frame_start) * 1000, 2)
    frame_data["cpu_percent"] = psutil.cpu_percent()
    frame_data["ram_usage_mb"] = round(psutil.virtual_memory().used / (1024 * 1024), 2)

    log.append(frame_data)
    frame_count += 1

    cv2.imshow("2-Stage Detection", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()

# Final metrics
total_time = time.time() - start_time
fps = round(frame_count / total_time, 2)

output = {
    "video_path": video_path,
    "total_frames": frame_count,
    "average_fps": fps,
    "frames": log
}

with open("inference_metrics.json", "w") as f:
    json.dump(output, f, indent=2)

print(" Saved 2-stage inference results to inference_metrics.json")
