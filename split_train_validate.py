import os
import shutil
import random
from pathlib import Path

def create_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def split_dataset_within_each_cxx(root_dir, train_ratio=0.8, image_exts=('.jpg', '.jpeg', '.png')):
    """
    For each `sXX/cXXX/`, split `img1/` and `label/` into local `training/` and `validate/` folders.
    Files will be moved instead of copied.
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
                print(f"    Skipping {c_folder} as one or both of img1 or label directories are missing.")
                continue

            print(f"    Found {len(os.listdir(img_dir))} images and {len(os.listdir(lbl_dir))} labels.")

            all_pairs = []
            for img_file in os.listdir(img_dir):
                if Path(img_file).suffix.lower() not in image_exts:
                    continue

                base_name = Path(img_file).stem
                img_path = os.path.join(img_dir, img_file)
                lbl_path = os.path.join(lbl_dir, base_name + '.txt')

                if os.path.exists(lbl_path):
                    all_pairs.append((img_path, lbl_path))
                else:
                    print(f"      Warning: No label file for {img_file}!")

            if not all_pairs:
                print(f"    No valid image-label pairs found in {c_folder}. Skipping.")
                continue

            print(f"    Found {len(all_pairs)} valid image-label pairs.")

            random.shuffle(all_pairs)
            split_idx = int(len(all_pairs) * train_ratio)
            train_pairs = all_pairs[:split_idx]
            val_pairs = all_pairs[split_idx:]

            # Create training/validate folders inside img1 and label
            img_train = create_dir(os.path.join(img_dir, 'training'))
            img_val = create_dir(os.path.join(img_dir, 'validate'))
            lbl_train = create_dir(os.path.join(lbl_dir, 'training'))
            lbl_val = create_dir(os.path.join(lbl_dir, 'validate'))

            print(f"    Creating directories: {img_train}, {img_val}, {lbl_train}, {lbl_val}")

            for img_src, lbl_src in train_pairs:
                shutil.move(img_src, os.path.join(img_train, os.path.basename(img_src)))
                shutil.move(lbl_src, os.path.join(lbl_train, os.path.basename(lbl_src)))
                print(f"      Moved {os.path.basename(img_src)} to {img_train}")
                print(f"      Moved {os.path.basename(lbl_src)} to {lbl_train}")

            for img_src, lbl_src in val_pairs:
                shutil.move(img_src, os.path.join(img_val, os.path.basename(img_src)))
                shutil.move(lbl_src, os.path.join(lbl_val, os.path.basename(lbl_src)))
                print(f"      Moved {os.path.basename(img_src)} to {img_val}")
                print(f"      Moved {os.path.basename(lbl_src)} to {lbl_val}")

            print(f"✅ {s_folder}/{c_folder}: {len(train_pairs)} training, {len(val_pairs)} validation samples")
# USAGE EXAMPLE
if __name__ == "__main__":
    dataset_root = "/home/luka/Desktop/AIC_2023_Track2/AIC23_Track2_NL_Retrieval/data/validation"  # CHANGE THIS TO YOUR PATH!
    split_dataset_within_each_cxx(dataset_root)