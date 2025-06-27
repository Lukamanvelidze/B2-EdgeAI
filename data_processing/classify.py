import os
import shutil

root_dir = "/path/to/root_dir"  # Change this to your root folder path
output_dir = "./classification_dataset"

def copy_classification_data():
    # Iterate over c6, c7, etc.
    for c_folder in os.listdir(root_dir):
        c_path = os.path.join(root_dir, c_folder)
        if not os.path.isdir(c_path):
            continue

        # Iterate over part1, part2, etc.
        for part_folder in os.listdir(c_path):
            part_path = os.path.join(c_path, part_folder)
            if not os.path.isdir(part_path):
                continue

            # Now inside partX/train or partX/val
            for split in ["train", "val"]:
                split_path = os.path.join(part_path, split)
                if not os.path.isdir(split_path):
                    continue

                out_split_dir = os.path.join(output_dir, split)
                os.makedirs(out_split_dir, exist_ok=True)

                # Process all images in this split folder
                for file in os.listdir(split_path):
                    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    img_path = os.path.join(split_path, file)
                    label_file = os.path.splitext(file)[0] + ".txt"
                    label_path = os.path.join(split_path, label_file)

                    if not os.path.isfile(label_path):
                        print(f"Warning: Label file {label_path} missing, skipping")
                        continue

                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    if len(lines) != 1:
                        print(f"Warning: More than one label in {label_path}, skipping")
                        continue

                    class_id = lines[0].split()[0]

                    # Make class folder in output dir
                    class_dir = os.path.join(out_split_dir, class_id)
                    os.makedirs(class_dir, exist_ok=True)

                    dst_path = os.path.join(class_dir, file)
                    shutil.copy2(img_path, dst_path)

                print(f"Finished processing {c_folder}/{part_folder}/{split}")

if __name__ == "__main__":
    copy_classification_data()
    print("All done!")
