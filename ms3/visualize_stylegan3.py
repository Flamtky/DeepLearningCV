import os
import re

def extract_seed(filename):
    match = re.search(r'seed(\d+)_tensor', filename)
    if match:
        return int(match.group(1))
    return None

def seed_to_filename(seed, label_folder_name):
    try:
        age, gen = map(int, list(label_folder_name))
    except ValueError:
        raise ValueError(f"Invalid label folder name (got {label_folder_name})")

    return f"seed{seed:04d}_tensor([[{age}., {gen}.]], device='cuda:0').png"

def preview_seed(seed: int):
    """
    Generate a grid preview image of a seed. X is the gender and Y is the age.
    """
    import cv2
    import numpy as np

    GEN_DIR = "./train/gen"
    SUBFOLDERS = os.listdir(GEN_DIR)
    SUBFOLDERS = [subfolder for subfolder in SUBFOLDERS if len(subfolder) == 2] # ignore other folders

    # Load images
    images = {}
    for subfolder in SUBFOLDERS:
        img_path = os.path.join(GEN_DIR, subfolder, seed_to_filename(seed, subfolder))
        img = cv2.imread(img_path)
        if img is not None:
            age, gen = map(int, list(subfolder))
            if age not in images:
                images[age] = {}
            images[age][gen] = img

    # Determine grid size
    max_age = max(images.keys())
    max_gen = max(max(gens.keys()) for gens in images.values())
    max_height = max(img.shape[0] for gens in images.values() for img in gens.values())
    max_width = max(img.shape[1] for gens in images.values() for img in gens.values())

    # Resize images
    for age in images:
        for gen in images[age]:
            images[age][gen] = cv2.resize(images[age][gen], (max_width, max_height))

    # Create grid
    grid_image = np.zeros((max_height * (max_age + 1), max_width * (max_gen + 1), 3), dtype=np.uint8)
    for age in images:
        for gen in images[age]:
            row = age
            col = gen
            grid_image[row * max_height:(row + 1) * max_height, col * max_width:(col + 1) * max_width, :] = images[age][gen]

    cv2.imwrite("seed_preview.png", grid_image)

if __name__ == "__main__":
    import argparse
    import random
    parser = argparse.ArgumentParser(description="Generate a grid preview image of a seed.")
    parser.add_argument("--seed", type=int, default=0, help="The seed to preview.")
    args = parser.parse_args()

    if args.seed == 0:
        args.seed = random.randint(1, 50000)
        print(f"Random seed generated: {args.seed}")

    preview_seed(args.seed)