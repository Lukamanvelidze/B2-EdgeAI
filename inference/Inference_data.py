import cv2
import time
import json
import psutil
from ultralytics import YOLO

# Load model
model = YOLO('global_model.pt')
model.model.names = {
    0: "bike", 1: "bus", 2: "caravan", 3: "coupe", 4: "crossover", 5: "hatchback",
    6: "jeep", 7: "mpv", 8: "pickup-truck", 9: "sedan", 10: "suv", 11: "taxi",
    12: "truck", 13: "van", 14: "vehicle", 15: "wagon"
}
label_map = model.model.names

video_path = '/home/luka/Desktop/AIC_2023_Track2/AIC23_Track2_NL_Retrieval/data/validation/S02/c007/vdo.avi'
delay = 1  # ms between frames
log = []

frame_count = 0
start_time = time.time()

try:
    # Start streaming inference generator
    results = model.predict(
        source=video_path,
        show=True,
        imgsz=416,
        conf=0.25,
        stream=True,
        verbose=False
    )
    
    # Initialize previous time before first result for inference timing
    prev_time = time.perf_counter()

    for i, result in enumerate(results):
        curr_time = time.perf_counter()
        inference_time = curr_time - prev_time
        prev_time = curr_time
        
        frame_data = {}
        frame_data["frame_id"] = i
        
        # Collect predictions
        predictions = []
        boxes = result.boxes
        if boxes is not None:
            for j in range(len(boxes)):
                cls_id = int(boxes.cls[j].item())
                conf = float(boxes.conf[j].item())
                xyxy = [float(x.item()) for x in boxes.xyxy[j]]
                
                predictions.append({
                    "class_id": cls_id,
                    "label": label_map[cls_id],
                    "confidence": round(conf, 4),
                    "bbox": xyxy  # [x1, y1, x2, y2]
                })
        
        frame_data["predictions"] = predictions
        
        # Corrected inference time measurement
        frame_data["inference_time_ms"] = round(inference_time * 1000, 2)
        
        # System usage
        frame_data["cpu_percent"] = psutil.cpu_percent()
        frame_data["ram_usage_mb"] = round(psutil.virtual_memory().used / (1024 * 1024), 2)
        
        log.append(frame_data)
        
        frame_count += 1
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print("Quitting...")
            break
    
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
    
    print(f"[✔] Saved metrics to inference_metrics.json")

except Exception as e:
    print(f"[✘] Inference failed: {e}")
    print("Try the manual method instead")
