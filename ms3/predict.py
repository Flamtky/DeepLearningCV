import os
import keras
import numpy as np
import cv2
import csv
from model import (
    AGE_GROUPS,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = (
    "./cur_best_model.keras"
)

def load_model_from_file():
    return keras.models.load_model(
        MODEL_PATH,
    )

def seed_to_filename(seed, label_folder_name):
    try:
        age, gen = map(int, list(label_folder_name))
    except ValueError:
        raise ValueError(f"Invalid label folder name (got {label_folder_name})")

    return f"seed{seed:04d}_tensor([[{age}., {gen}.]], device='cuda:0').png"

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads and preprocesses an image for prediction.
    """
    # Read and preprocess the image
    img = keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img


def load_ground_truth(csv_file="./test/faces_anno_test.csv"):
    ground_truth = {}
    with open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            ground_truth[row["image_path"]] = {
                "age_group": row["age_group"],
                "gender": row["gender"],
            }
    return ground_truth


def predict_image(image_path, ground_truth):
    """
    Predicts face presence, age group, and gender for a given image.
    """
    model = load_model_from_file()

    # Preprocess
    img = preprocess_image(image_path)

    predictions = model.predict(img)
    pred_face, pred_age, pred_gender = predictions.values()

    age_index = np.argmax(pred_age[0])
    gender_index = np.argmax(pred_gender[0])

    age_text = AGE_GROUPS[age_index]
    gender_text = "Male" if gender_index == 0 else "Female"

    # Ground truth
    image_name = os.path.basename(image_path)
    if ground_truth is None:
        gt_info = {"age_group": "Unknown", "gender": "Unknown"}
    else:
        gt_info = ground_truth.get(
            f"faces/{image_name}", {"age_group": "Unknown", "gender": "Unknown"}
        )
    gt_age = gt_info["age_group"]
    gt_gender = gt_info["gender"]

    if pred_face[0][0] > 0.5:
        return

    print(f"Image: {image_name}")
    print("Prediction Results:")
    print(f"  Face present probability: {pred_face[0][0]:.2f}")
    print(f"  Predicted Age Group: {age_text}")
    print(f"  Predicted Gender: {gender_text}")
    print(f'  Raw Predictions: {predictions}')
    print("Ground Truth:")
    print(f"  Age Group: {gt_age}")
    print(f"  Gender: {gt_gender}")

    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'")
        return

    # Write text onto the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        image, f"Age: {age_text}", (10, 30), font, 1, (0, 0, 255), 2
    )  # Red color
    cv2.putText(image, f"Gender: {gender_text}", (10, 70), font, 1, (0, 0, 255), 2)

    # Display the image
    cv2.imwrite("prediction.jpg", image)
    input("Press Enter to continue...")


SUBFOLDER = "10"

def predict_seed(seed, label_folder_name):
    IMAGE_PATH = os.path.join(f"./train/gen/{SUBFOLDER}", seed_to_filename(seed, label_folder_name))
    predict_image(IMAGE_PATH, None)

if __name__ == "__main__":

    # predict_seed(736, SUBFOLDER)

    IMAGES = "./val/gen/01"
    ground_truth = load_ground_truth()

    for image_name in os.listdir(IMAGES):
        IMAGE_PATH = os.path.join(IMAGES, image_name)

        if not os.path.exists(IMAGE_PATH):
            print(f"Error: Image file '{IMAGE_PATH}' not found")
        else:
            predict_image(IMAGE_PATH, ground_truth)
