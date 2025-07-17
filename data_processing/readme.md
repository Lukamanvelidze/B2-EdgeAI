###  Data Preprocessing and Challenges

During the project, several challenges arose — especially in transforming the dataset into a format that was both **semantically** and **syntactically** compatible with our training pipeline.  
To address these issues, we developed several custom scripts, each designed to solve a specific part of the preprocessing and labeling workflow:

| Script                 | Description |
|------------------------|-------------|
| `data_partitioning.py` | Splits the original dataset into training, validation, sets for federated learning experiments. |
| `crop.py`              | Crops vehicle regions from images based on bounding box annotations. These cropped images are then used to train a **vehicle classifier**. |
| `convert_to_yolo.py`   | Converts the existing annotations into YOLO format, making them suitable for training YOLO11n models. |
| `classify.py`          | Uses our custom-trained classifier to label the cropped vehicle images with more specific vehicle types (e.g., `sedan`, `SUV`, `truck`). This step enriched the dataset by annotating vehicles that were originally missing labels. |
| `pipeline.py`          | Orchestrates the entire data preparation pipeline — from reading the dataset to generating the final YOLO-compatible, fully annotated dataset. |

>  This toolchain allowed us to go from a **partially annotated dataset** (with only 1–2 vehicles labeled per image) to a **fully labeled dataset**, significantly improving the quality of training and evaluation.

