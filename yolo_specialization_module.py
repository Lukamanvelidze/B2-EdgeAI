import math
import numpy as np
import cv2
import json
from sklearn.cluster import KMeans
from collections import defaultdict
from sort.sort import Sort
from ultralytics import YOLO 


def compute_direction_speed(track_points, fps):
    directions, speeds = [], []
    for i in range(1, len(track_points)):
        dx = track_points[i][1] - track_points[i-1][1]
        dy = track_points[i][2] - track_points[i-1][2]
        direction = math.degrees(math.atan2(-dy, dx)) % 360
        speed = math.hypot(dx, dy) * fps
        directions.append(direction)
        speeds.append(speed)
    return directions, speeds

def get_dominant_color(image_crop):
    image_crop = cv2.resize(image_crop, (50, 50))
    image_crop = image_crop.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, n_init='auto').fit(image_crop)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    return dominant_color  # BGR

def map_to_color_name(bgr):
    b, g, r = bgr
    if r > 150 and g < 100 and b < 100:
        return "red"
    elif g > 150 and r < 100 and b < 100:
        return "green"
    elif b > 150 and r < 100 and g < 100:
        return "blue"
    elif r > 150 and g > 150 and b > 150:
        return "white"
    elif r < 80 and g < 80 and b < 80:
        return "black"
    else:
        return "unknown"

def generate_metadata_entry(class_id, bbox, speed, direction, color):
    return {
        "class_id": class_id,
        "bbox": bbox,
        "speed": speed,
        "direction": direction,
        "color": color
    }


def save_metadata_to_json(metadata_list, output_path):
    with open(output_path, 'w') as f:
        json.dump({"objects": metadata_list}, f, indent=4)


if __name__ == "__main__":
    cap = cv2.VideoCapture("video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = Sort()
    tracks = defaultdict(list)
    frame_id = 0
    metadata = []

    model = YOLO("yolov11.pt") 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

        dets = np.array(detections)
        tracked_objects = tracker.update(dets)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            tracks[track_id].append((frame_id, cx, cy))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            color = get_dominant_color(crop)
            color_name = map_to_color_name(color)

            if len(tracks[track_id]) >= 2:
                dir_list, spd_list = compute_direction_speed(tracks[track_id], fps)
                direction = dir_list[-1]
                speed = spd_list[-1]
                norm_bbox = [cx / frame.shape[1], cy / frame.shape[0], (x2 - x1) / frame.shape[1], (y2 - y1) / frame.shape[0]]
                entry = generate_metadata_entry(0, norm_bbox, speed, direction, color_name)
                metadata.append(entry)

        frame_id += 1

    cap.release()
    save_metadata_to_json(metadata, "tracked_output.json")

