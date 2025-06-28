import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# this way we will emulate 2 - stage object detection to generate the new dataset for our use-case scenario and further fine tune it

# === Paths ===
IMAGE_FOLDER = "full_images"             # folder with full images (not cropped)
OUTPUT_LABEL_DIR = "new_labels"          # where new YOLO-format labels will go
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# === Load Models ===
base_model = YOLO("yolov11n.pt")           # base model to detect bounding boxes
fine_model = YOLO("global_model.pt")      # fine-grained classifier (your model)

# Define  fine class names (same as what you used during fine-tuning)
fine_model.model.names = {
    0: "bike",
    1: "bus",
    2: "caravan",
    3: "coupe",
    4: "crossover",
    5: "hatchback",
    6: "jeep",
    7: "mpv",
    8: "pickup-truck",
    9: "sedan",
    10: "suv",
    11: "taxi",
    12: "truck",
    13: "van",
    14: "vehicle",
    15: "wagon"
}

# === Collect image filenames ===
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".jpg", ".png"))]

# === Process each image ===
for img_name in tqdm(image_files, desc="Generating new labels"):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        continue

    height, width, _ = img.shape
    try:
        base_results = base_model(img, conf=0.4)[0]
    except Exception as e:
        print(f"Base model failed on {img_name}: {e}")
        continue

    yolo_lines = []

    for box in base_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        try:
            fine_results = fine_model(crop, conf=0.25)[0]
        except Exception as e:
            print(f"Fine model failed on crop in {img_name}: {e}")
            continue

        if not fine_results.boxes:
            continue

        # Assume best box is first box ( classifier should output one box ideally)
        fine_box = fine_results.boxes[0]
        fine_class_id = int(fine_box.cls[0])

        # Convert bounding box to YOLO format (relative coords)
        x_center = (x1 + x2) / 2.0 / width
        y_center = (y1 + y2) / 2.0 / height
        w_norm = (x2 - x1) / width
        h_norm = (y2 - y1) / height
        yolo_lines.append(f"{fine_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Save .txt file with same base name as image
    if yolo_lines:
        label_filename = img_name.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(OUTPUT_LABEL_DIR, label_filename)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
