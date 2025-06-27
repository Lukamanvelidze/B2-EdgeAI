import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load both YOLO models
detector = YOLO("yolo_base.pt")      # good at detecting boxes
classifier = YOLO("modelc7.pt")      # your custom model, good at classifying

video_path = "/your/path/video.avi"
delay = 50

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get bounding boxes from base YOLO
    results = detector(frame, conf=0.4, iou=0.4)
    result = results[0]
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Crop the object from the original frame
            cropped = frame[y1:y2, x1:x2]
            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                continue

            # Run the classifier YOLO model on the crop
            classifier_result = classifier.predict(
                source=cropped,  # just a small image
                imgsz=224,       # or size used in your training
                conf=0.1,        # allow even weak predictions
                iou=0.5,
                verbose=False,
                stream=False
            )[0]

            # Get the most confident prediction from the classifier
            if classifier_result.boxes is not None and len(classifier_result.boxes) > 0:
                best_box = max(classifier_result.boxes, key=lambda b: b.conf)
                class_id = int(best_box.cls[0].item())
                label = classifier.model.names[class_id]
            else:
                label = "Unknown"

            # Draw on original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("YOLO Hybrid Inference", frame)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
