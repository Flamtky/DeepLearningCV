import os
import shutil
import random
import csv
from multiprocessing import Pool, cpu_count


def create_directories(base_dir):
    for dataset_type in ["train", "val", "test"]:
        for category in ["faces", "no_faces"]:
            dir_path = os.path.join(base_dir, dataset_type, category)
            os.makedirs(dir_path, exist_ok=True)


def split_data(file_list, train_ratio, val_ratio, test_ratio):
    random.shuffle(file_list)
    total = len(file_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    test_end = val_end + int(total * test_ratio)
    train_files = file_list[:train_end]
    val_files = file_list[train_end:val_end]
    test_files = file_list[val_end:test_end]
    return train_files, val_files, test_files


def copy_file(args):
    src, dst = args
    shutil.copy2(src, dst)


def process_files(file_list, src_dir, dst_dir):
    args = []
    for filename in file_list:
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dst_dir, filename)
        args.append((src, dst))
    with Pool(cpu_count()) as pool:
        pool.map(copy_file, args)


def split_csv(csv_file, train_files, val_files, test_files, base_dir):
    with open(csv_file, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile, delimiter=";")
        fieldnames = reader.fieldnames
        data = list(reader)

    datasets = {
        "train": set(train_files),
        "val": set(val_files),
        "test": set(test_files),
    }

    for dataset_type in ["train", "val", "test"]:
        out_csv = os.path.join(base_dir, dataset_type, f"faces_anno_{dataset_type}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                image_path = row["image_path"].replace("faces/", "")
                if image_path in datasets[dataset_type]:
                    # Update the image path to reflect new location
                    row["image_path"] = os.path.join("faces", image_path)
                    writer.writerow(row)


def main():
    base_dir = "."
    faces_dir = os.path.join(base_dir, "faces")
    no_faces_dir = os.path.join(base_dir, "no_faces")
    csv_file = os.path.join(base_dir, "faces_anno.csv")

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    create_directories(base_dir)

    # Get list of image files
    faces_files = os.listdir(faces_dir)
    no_faces_files = os.listdir(no_faces_dir)

    # Split the data
    faces_train, faces_val, faces_test = split_data(
        faces_files, train_ratio, val_ratio, test_ratio
    )
    no_faces_train, no_faces_val, no_faces_test = split_data(
        no_faces_files, train_ratio, val_ratio, test_ratio
    )

    # Process faces images
    process_files(faces_train, faces_dir, os.path.join(base_dir, "train", "faces"))
    process_files(faces_val, faces_dir, os.path.join(base_dir, "val", "faces"))
    process_files(faces_test, faces_dir, os.path.join(base_dir, "test", "faces"))

    # Process no_faces images
    process_files(
        no_faces_train, no_faces_dir, os.path.join(base_dir, "train", "no_faces")
    )
    process_files(no_faces_val, no_faces_dir, os.path.join(base_dir, "val", "no_faces"))
    process_files(
        no_faces_test, no_faces_dir, os.path.join(base_dir, "test", "no_faces")
    )

    # Split CSV file for faces
    split_csv(csv_file, faces_train, faces_val, faces_test, base_dir)

def split_data(src_dir, target_dir, split_ratio=0.2):
    data_files = os.listdir(src_dir)
    split_count = int(len(data_files) * split_ratio)
    random.seed(42)
    random.shuffle(data_files)
    files_to_move = data_files[:split_count]

    os.makedirs(target_dir, exist_ok=True)
    process_files(files_to_move, src_dir, target_dir)

if __name__ == "__main__":
    #main()
    #split_data("../train/lhq_256", "../train/1lhq_256", split_ratio=0.9)
    split_data("../stylegan3/out/00", "../val/gen/00", split_ratio=0.1)
    split_data("../stylegan3/out/10", "../val/gen/10", split_ratio=0.1)
    split_data("../stylegan3/out/01", "../val/gen/01", split_ratio=0.1)
    split_data("../stylegan3/out/11", "../val/gen/11", split_ratio=0.1)
