import os
import shutil
import random
from pathlib import Path

def create_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def partition_and_copy(pairs, base_folder, num_parts, split_type):
    """
    Partition the given (image, label) pairs into N parts and copy them into the correct directory structure.
    Each part has a train or val folder with img and label inside.
    """
    part_size = len(pairs) // num_parts
    extras = len(pairs) % num_parts
    start_idx = 0

    for i in range(num_parts):
        end_idx = start_idx + part_size + (1 if i < extras else 0)
        part_pairs = pairs[start_idx:end_idx]

        part_folder = create_dir(os.path.join(base_folder, f'part{i+1}', split_type))
        img_dir = create_dir(os.path.join(part_folder, 'img'))
        lbl_dir = create_dir(os.path.join(part_folder, 'label'))

        for img_src, lbl_src in part_pairs:
            shutil.copy2(img_src, os.path.join(img_dir, os.path.basename(img_src)))
            shutil.copy2(lbl_src, os.path.join(lbl_dir, os.path.basename(lbl_src)))

        print(f"    ✅ {split_type} Partition part{i+1}: {len(part_pairs)} samples")
        start_idx = end_idx


def split_dataset_within_each_cxx(root_dir, train_ratio=0.8, num_partitions=2, image_exts=('.jpg', '.jpeg', '.png')):
    """
    For each `sXX/cXXX/`, split `img1/` and `labels/` into training and validation,
    then partition them into multiple parts under the `dataset/` folder.
    """
    for s_folder in sorted(os.listdir(root_dir)):
        s_path = os.path.join(root_dir, s_folder)
        if not os.path.isdir(s_path):
            print(f"Skipping non-directory {s_path}")
            continue

        print(f"Processing sequence folder: {s_folder}")

        for c_folder in sorted(os.listdir(s_path)):
            c_path = os.path.join(s_path, c_folder)
            if not os.path.isdir(c_path):
                print(f"Skipping non-directory {c_path}")
                continue

            print(f"  Processing category folder: {c_folder}")

            img_dir = os.path.join(c_path, 'img1')
            lbl_dir = os.path.join(c_path, 'labels')

            if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
                print(f"    Skipping {c_folder} as one or both of img1 or labels directories are missing.")
                continue

            all_pairs = []
            for img_file in os.listdir(img_dir):
                print(img_file)
                if Path(img_file).suffix.lower() not in image_exts:
                    continue

                base_name = Path(img_file).stem.lower()  # normalize base name
                img_path = os.path.join(img_dir, img_file)
                lbl_path = os.path.join(lbl_dir, base_name + '.txt')

                if os.path.exists(lbl_path):
                    all_pairs.append((img_path, lbl_path))
                else:
                    print(f"      Warning: No label file for {img_file}!")

            if not all_pairs:
                print(f"    No valid image-label pairs found in {c_folder}. Skipping.")
                continue

            random.shuffle(all_pairs)
            split_idx = int(len(all_pairs) * train_ratio)
            train_pairs = all_pairs[:split_idx]
            val_pairs = all_pairs[split_idx:]

            # Dataset root folder
            dataset = create_dir(os.path.join(c_path, 'dataset'))

            # Partition both train and val sets
            partition_and_copy(train_pairs, dataset, num_partitions, split_type='train')
            partition_and_copy(val_pairs, dataset, num_partitions, split_type='val')

            print(f"✅ {s_folder}/{c_folder}: {len(train_pairs)} training, {len(val_pairs)} validation samples into {num_partitions} partitions")

if __name__ == "__main__":
    dataset_root = "/home/ado/data/validation"
    split_dataset_within_each_cxx(dataset_root, train_ratio=0.8, num_partitions=2)



"""
folder structure
dataset/
├── part1/
│   ├── train/
│   │   ├── img/
│   │   └── label/
│   └── val/
│       ├── img/
│       └── label/
├── part2/
│   ├── train/
│   │   ├── img/
│   │   └── label/
│   └── val/
│       ├── img/
│       └── label/
...

"""
