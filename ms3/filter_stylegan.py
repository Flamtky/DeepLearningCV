import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # First GPU

MODEL_PATH = "cur_best_model.keras"
GEN_DIR = "./train/gen"
OUTPUT_FILE = "no_face_seeds.txt"

def load_model():
    import keras
    import model as mo
    model: keras.Model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "face_loss": mo.face_loss,
            "conditional_age_loss": mo.conditional_age_loss,
            "conditional_gender_loss": mo.conditional_gender_loss,
            "age_acc": mo.age_acc,
            "gen_acc": mo.gen_acc,
        },
    )
    return model

def preprocess_image(img_path, target_size=(224, 224)):
    import tensorflow as tf
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0 # Normalize
    img = tf.expand_dims(img, axis=0) # Batch dimension
    return img

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

def gen_invalid_seeds_file():
    model = load_model()
    no_face_seeds = []
    for subfolder in ["10", "11"]:
        folder_path = os.path.join(GEN_DIR, subfolder)
        for fname in os.listdir(folder_path):
            if fname.endswith(".png"):
                seed = extract_seed(fname)
                if seed is not None:
                    img_path = os.path.join(folder_path, fname)
                    img_array = preprocess_image(img_path)
                    predictions = model.predict(img_array)
                    face_prob = predictions["face"][0][0]
                    print(f"Seed {seed} face prob: {face_prob} (raw: {predictions}")
                    if face_prob < 0.5:
                        no_face_seeds.append(seed)

    # Remove duplicates and sort
    no_face_seeds = sorted(set(no_face_seeds))
    with open(OUTPUT_FILE, "w") as f:
        for seed in no_face_seeds:
            f.write(f"{seed}\n")

def preview_no_faces():
    """
    Generate a grid preview image with all the seeds that have no face detected and saves it to a file.
    """
    import cv2
    import numpy as np
    with open("no_face.txt", "r") as f:
        no_face_seeds = [int(line.strip()) for line in f.readlines()]

    # Load images
    images = []
    for seed in no_face_seeds:
        img_path = os.path.join(GEN_DIR, "10", seed_to_filename(seed, "10"))
        img = cv2.imread(img_path)
        # Add seed text with black outline
        text = str(seed)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = 30
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        images.append(img)

    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(len(images))))
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    # Resize images
    for i, img in enumerate(images):
        images[i] = cv2.resize(img, (max_width, max_height))

    # Create grid
    grid_image = np.zeros((max_height * grid_size, max_width * grid_size, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        grid_image[row * max_height:(row + 1) * max_height, col * max_width:(col + 1) * max_width, :] = img

    cv2.imwrite("no_face_preview.png", grid_image)

def filter_no_faces():
    """
    Filter out the found no face seeds from the dataset. By moving the images to a different folder.
    """
    with open("no_face.txt", "r") as f:
        no_face_seeds = [int(line.strip()) for line in f.readlines()]

    NO_FACES_DIR = "./no_faces_stylegan"
    SUBFOLDERS = os.listdir(GEN_DIR)
    SUBFOLDERS = [subfolder for subfolder in SUBFOLDERS if len(subfolder) == 2] # ignore other folders

    for seed in no_face_seeds:
        image_c = 0
        for subfolder in SUBFOLDERS:
            img_name = seed_to_filename(seed, subfolder)
            img_path = os.path.join(GEN_DIR, subfolder, img_name)
            if os.path.exists(img_path):
                image_c += 1
                new_path = os.path.join(NO_FACES_DIR, f'{seed:04d}_{subfolder}.png')
                if os.path.exists(new_path):
                    append = 0
                    while os.path.exists(new_path + append):
                        append += 1
                    new_path += append
                os.rename(img_path, new_path)
            else: 
                print(f"Warning: {img_path} not found", end="")
        print(f"Moved seed {seed} to {new_path} ({image_c} images)")

if __name__ == "__main__":
    filter_no_faces()