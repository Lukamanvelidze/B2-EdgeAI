import json
import os
import cv2

#when u run this code be in data folder

# Define the vehicle class mapping and canonical class list (using set)
vehicle_map = {
    "wagon": "wagon",
    "sedan": "sedan",
    "suv": "suv",
    "car": "vehicle",
    "chevrolet": "vehicle",
    "honda": "vehicle",
    "caravan": "caravan",
    "bike": "bike",
    "toyota": "vehicle",
    "vehicle": "vehicle",
    "van": "van",
    "hatchback": "hatchback",
    "mpv": "mpv",
    "mini cooper": "coupe",
    "mercedes": "vehicle",
    "jeep": "jeep",
    "cherokee": "jeep",
    "minivan": "van",
    "pickup truck": "pickup-truck",
    "pickup-truck": "pickup-truck",
    "truck": "truck",
    "tractor-trailer": "truck",
    "semi-truck": "truck",
    "pickup": "pickup-truck",
    "pick up": "pickup-truck",
    "bus": "bus",
    "cross-over": "crossover",
    "cross over": "crossover",
    "crossover": "crossover",
    "cross": "crossover",
    "sports": "vehicle",
    "subaru": "vehicle",
    "taxi": "taxi",
    "coup": "coupe",
    "coupe": "coupe",
    "couple": "coupe",  # often an OCR mistake
    "cargo truck": "pickup-truck",
    "Chevvy": "vehicle",
    "spv": "suv",
    "svu": "suv",
    "ford mustang": "vehicle",
    "chevy": "vehicle",
    "audi": "vehicle",
    "sede": "sedan",  # typo
    "hatckback": "hatchback",  # typo
    "cargo pickup truck": "pickup-truck",
    "can": "car"  # possibly bad OCR
}

# Define canonical class list , calss_id corresponds to the position in the list
canonical_classes = [
    "bike",          # 0
    "bus",           # 1
    "caravan",       # 2
    "coupe",         # 3
    "crossover",     # 4
    "hatchback",     # 5
    "jeep",          # 6
    "mpv",           # 7
    "pickup-truck",  # 8
    "sedan",         # 9
    "suv",           # 10
    "taxi",          # 11
    "truck",         # 12
    "van",           # 13
    "vehicle",       # 14
    "wagon"          # 15
]

# Define a mapping for class names to class indexes based on set order
class_index_mapping = {class_name: index for index, class_name in enumerate(canonical_classes)}


with open("train-tracks.json", "r") as f:
    data = json.load(f)


def extract_class(nl_description, vehicle_map):
    tokens = nl_description.lower().replace(".", "").split()
    print(f"Tokens: {tokens}")
    for token in tokens:
        if token in vehicle_map:
            mapped_class = vehicle_map[token]
            class_id = class_index_mapping.get(mapped_class, "NOT_FOUND")
            print(f"  → Matched token: '{token}' → Class: '{mapped_class}' (ID: {class_id})")
            return vehicle_map[token]
    return "vehicle"  # default


for uuid, entry in data.items():
    frames = entry["frames"]
    boxes = entry["boxes"]
    nl_descriptions = entry["nl"]  # All NL descriptions for this sequence

    # Extract class candidates from all NL descriptions
    class_candidates = []
    for nl_desc in nl_descriptions:
        class_name = extract_class(nl_desc, vehicle_map)
        class_candidates.append(class_name)
        print(f"NL: '{nl_desc}' → Class: {class_name}")  # Debug

    # Choose the most frequent class (majority vote)
    from collections import Counter

    class_counter = Counter(class_candidates)
    most_common_class = class_counter.most_common(1)[0][0]

    print(f"Final class for sequence: {most_common_class}")  # Debug

    # Get class ID
    try:
        class_id = class_index_mapping[most_common_class]
    except KeyError:
        print(f"Class '{most_common_class}' not found. Using 'vehicle' (ID 14).")
        class_id = 14  # Default to "vehicle"

    for frame_path, box in zip(frames, boxes):
        # Get image dimensions
        if not os.path.exists(frame_path):
            print(f"Image {frame_path} not found. Skipping.")
            continue

        # Read image to get width/height
        img = cv2.imread(frame_path)
        height, width, _ = img.shape

        # Convert box to YOLO format
        x, y, w, h = box
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w_norm = w / width
        h_norm = h / height

        # Prepare YOLO line
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

        # Create labels folder path
        frame_dir = os.path.dirname(frame_path)  # e.g., "./validation/S02/c006/img1"
        parent_dir = os.path.dirname(frame_dir)  # e.g., "./validation/S02/c006"
        labels_dir = os.path.join(parent_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        # Write label file
        frame_name = os.path.basename(frame_path)  # e.g., "000001.jpg"
        label_name = frame_name.replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, label_name)

        with open(label_path, "w") as f:
            f.write(yolo_line + "\n")

print("Conversion complete!")
















