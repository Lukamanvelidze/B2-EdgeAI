import cv2
from ultralytics import YOLO

# Base YOLO model for bounding box detection
detector = YOLO("yolov11n.pt")

# Custom YOLO classifier (your vehicle classifier)
classifier = YOLO("modelc7.pt")

# Optional: Override names for classifier if needed
classifier.model.names = {
    0: "bike", 1: "bus", 2: "caravan", 3: "coupe", 4: "crossover",
    5: "hatchback", 6: "jeep", 7: "mpv", 8: "pickup-truck", 9: "sedan",
    10: "suv", 11: "taxi", 12: "truck", 13: "van", 14: "vehicle", 15: "wagon"
}

# Filter only transport class IDs from base detector
transport_classes = {"car", "bicycle", "bus", "truck", "motorcycle"}
target_class_ids = {
    class_id for class_id, name in detector.model.names.items()
    if name.lower() in transport_classes
}
print("Detecting bounding boxes for class IDs:", target_class_ids)

# Inference video
video_path = "/path/to/video.avi"
cap = cv2.VideoCapture(video_path)
delay = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.predict(source=frame, imgsz=416, conf=0.25, stream=False)
    result = results[0]

    for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
        class_id = int(cls_id.item())
        if class_id not in target_class_ids:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        cropped = frame[y1:y2, x1:x2]

        # Predict class using custom YOLO classifier
        try:
            class_result = classifier.predict(source=cropped, imgsz=224, conf=0.3, stream=False, verbose=False)[0]
            if len(class_result.boxes) > 0:
                # Take top box (highest conf) only
                best_box = class_result.boxes[0]
                pred_cls = int(best_box.cls.item())
                label = classifier.model.names[pred_cls]
            else:
                label = "unknown"
        except Exception as e:
            label = "error"
            print(f"Classification error: {e}")

        # Draw detection bounding box (from base model), label from classifier
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Custom Classification on Detected Vehicles", frame)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
