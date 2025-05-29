import cv2
import math
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from sort import Sort


def get_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def compute_speed(prev, curr, time_elapsed_sec):
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    return math.hypot(dx, dy) / time_elapsed_sec


def compute_angle(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return math.degrees(math.atan2(-dy, dx)) % 360


def angle_to_direction(angle):
    if 337.5 <= angle or angle < 22.5:
        return "East"
    elif 22.5 <= angle < 67.5:
        return "North-East"
    elif 67.5 <= angle < 112.5:
        return "North"
    elif 112.5 <= angle < 157.5:
        return "North-West"
    elif 157.5 <= angle < 202.5:
        return "West"
    elif 202.5 <= angle < 247.5:
        return "South-West"
    elif 247.5 <= angle < 292.5:
        return "South"
    elif 292.5 <= angle < 337.5:
        return "South-East"


def map_rgb_to_color(r, g, b):
    if r > 150 and g > 150 and b > 150 and abs(r - g) < 30 and abs(g - b) < 30:
        return "white"
    elif r > 100 and g < 80 and b < 80:
        return "dark red"
    elif r > 100 and r > g and r > b:
        return "red"
    elif r < 80 and g < 80 and b < 80:
        return "black"
    elif abs(r - g) < 20 and abs(g - b) < 20 and r < 150:
        return "gray"
    elif r > 160 and g > 120 and b < 130:
        return "orange"
    elif r > 160 and g > 160 and b < 130:
        return "yellow"
    elif g > 120 and r < 130 and b < 130:
        return "green"
    elif g > 120 and b > 120 and r < 130:
        return "cyan"
    elif b > 120 and r < 130 and g < 130:
        return "blue"
    elif r > 120 and b > 120 and g < 100 and abs(r - b) < 30:
        return "purple"
    elif r > 160 and g > 80 and b > 120:
        return "pink"
    else:
        max_val = max(r, g, b)
        if max_val > 150:
            return "white"
        elif max_val == r:
            return "red"
        elif max_val == g:
            return "green"
        elif max_val == b:
            return "blue"
        return "unknown"


def get_average_rgb(crop):
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    resized = cv2.resize(crop, (50, 50))
    avg_rgb = np.mean(resized.reshape(-1, 3), axis=0)
    b, g, r = avg_rgb.astype(int)
    return r, g, b


# Setup
model = YOLO("yolov8n.pt")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture("vidoe2.webm")
if not cap.isOpened():
    print("ERROR: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps
direction_update_interval = int(fps * 2)
speed_update_interval = 10

track_history = defaultdict(list)
last_directions = {}
vehicle_colors = {}
vehicle_types = {}
vehicle_speeds = {}
last_speed_update = {}

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []
    temp_types = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label not in ["car", "truck", "bus"]:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append([x1, y1, x2, y2, conf])
        temp_types.append(label)

    dets = np.array(detections)
    tracks = tracker.update(dets)

    for i, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = map(int, track)
        center = get_center(x1, y1, x2, y2)
        track_history[track_id].append((frame_id, center))

        if track_id not in vehicle_types and i < len(temp_types):
            vehicle_types[track_id] = temp_types[i]

        if track_id not in vehicle_colors:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                r, g, b = get_average_rgb(crop)
                color = map_rgb_to_color(r, g, b)
                vehicle_colors[track_id] = color

        history = track_history[track_id]
        if len(history) >= 2:
            _, prev = history[-2]
            _, curr = history[-1]

            if (track_id not in last_directions or
                    frame_id - last_directions[track_id][0] > direction_update_interval):
                angle = compute_angle(prev, curr)
                direction = angle_to_direction(angle)
                last_directions[track_id] = (frame_id, direction)

            if (track_id not in last_speed_update or
                    frame_id - last_speed_update[track_id] >= speed_update_interval):
                elapsed_time = (frame_id - history[-2][0]) * frame_time
                if elapsed_time > 0:
                    speed = compute_speed(prev, curr, elapsed_time)
                    vehicle_speeds[track_id] = speed
                    last_speed_update[track_id] = frame_id

        direction = last_directions.get(track_id, (None, ""))[1]
        color = vehicle_colors.get(track_id, "")
        vehicle_type = vehicle_types.get(track_id, "")
        speed = vehicle_speeds.get(track_id, 0)

        label_text = f"{vehicle_type} / {direction} / {color} / {speed:.1f}px/s"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_size[0], y1), (255, 255, 255), -1)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    cv2.imshow("Vehicle Type / Direction / Color / Speed", frame)
    if cv2.waitKey(1) == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
