import json
import os
import cv2
from collections import Counter

vehicle_map = {
    "wagon": "wagon", "sedan": "sedan", "suv": "suv", "car": "vehicle",
    "chevrolet": "vehicle", "honda": "vehicle", "caravan": "caravan", "bike": "bike",
    "toyota": "vehicle", "vehicle": "vehicle", "van": "van", "hatchback": "hatchback",
    "mpv": "mpv", "mini cooper": "coupe", "mercedes": "vehicle", "jeep": "jeep",
    "cherokee": "jeep", "minivan": "van", "pickup truck": "pickup-truck",
    "pickup-truck": "pickup-truck", "truck": "truck", "tractor-trailer": "truck",
    "semi-truck": "truck", "pickup": "pickup-truck", "pick up": "pickup-truck",
    "bus": "bus", "cross-over": "crossover", "cross over": "crossover",
    "crossover": "crossover", "cross": "crossover", "sports": "vehicle",
    "subaru": "vehicle", "taxi": "taxi", "coup": "coupe", "coupe": "coupe",
    "couple": "coupe", "cargo truck": "pickup-truck", "Chevvy": "vehicle",
    "spv": "suv", "svu": "suv", "ford mustang": "vehicle", "chevy": "vehicle",
    "audi": "vehicle", "sede": "sedan", "hatckback": "hatchback",
    "cargo pickup truck": "pickup-truck", "can": "car"
}

canonical_classes = [
    "bike", "bus", "caravan", "coupe", "crossover", "hatchback", "jeep",
    "mpv", "pickup-truck", "sedan", "suv", "taxi", "truck", "van", "vehicle", "wagon"
]
class_index_mapping = {class_name: index for index, class_name in enumerate(canonical_classes)}

def extract_class(nl_description, vehicle_map):
    tokens = nl_description.lower().replace(".", "").split()
    for token in tokens:
        if token in vehicle_map:
            return vehicle_map[token]
    return "vehicle"

with open("train-tracks.json", "r") as f:
    data = json.load(f)

for uuid, entry in data.items():
    frames = entry["frames"]
    boxes = entry["boxes"]
    nl_descriptions = entry["nl"]

    class_candidates = [extract_class(desc, vehicle_map) for desc in nl_descriptions]
    class_counter = Counter(class_candidates)
    most_common_class = class_counter.most_common(1)[0][0]
    class_id = class_index_mapping.get(most_common_class, 14)

    for i, (frame_path, box) in enumerate(zip(frames, boxes)):
        if not os.path.exists(frame_path):
            print(f"[WARNING] Frame {frame_path} not found, skipping.")
            continue

        img = cv2.imread(frame_path)
        if img is None:
            print(f"[ERROR] Could not read image: {frame_path}")
            continue

        height, width, _ = img.shape

        # Crop bounding box coordinates (for cropping only)
        x, y, w, h = box
        crop_x1 = max(0, int(x))
        crop_y1 = max(0, int(y))
        crop_x2 = min(width, int(x + w))
        crop_y2 = min(height, int(y + h))
        cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

        if cropped_img.size == 0:
            print(f"[WARNING] Empty crop in {frame_path}, skipping.")
            continue

        frame_dir = os.path.dirname(frame_path)
        parent_dir = os.path.dirname(frame_dir)

        # Write YOLO label for full image with fixed coordinates
        labels_dir = os.path.join(parent_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        frame_name = os.path.basename(frame_path)
        label_name = frame_name.replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, label_name)
        with open(label_path, "w") as f:
            # fixed box: center 0.5 0.5, width=1, height=1
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

        # Save cropped image
        cropped_dir = os.path.join(parent_dir, "cropped")
        os.makedirs(cropped_dir, exist_ok=True)
        cropped_filename = f"{uuid}_{i}.jpg"
        cropped_path = os.path.join(cropped_dir, cropped_filename)
        cv2.imwrite(cropped_path, cropped_img)

        # Write label for cropped image with fixed coordinates
        cropped_labels_dir = os.path.join(parent_dir, "cropped_labels")
        os.makedirs(cropped_labels_dir, exist_ok=True)
        cropped_label_name = cropped_filename.replace(".jpg", ".txt")
        cropped_label_path = os.path.join(cropped_labels_dir, cropped_label_name)
        with open(cropped_label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

        print(f"[INFO] Saved cropped image and label: {cropped_path}, {cropped_label_path}")

print("Cropping and label generation complete.")
