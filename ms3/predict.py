import os
import keras
import numpy as np
import cv2
import csv
from model import (
    AGE_GROUPS,
)

MODEL_PATH = (
    "./models/0_efficientnetb0_relu_ds256_dl2_nobn_do0.0_wd0.0_lr1e-05_decay0.9.keras"
)


def load_model_from_file():
    return keras.models.load_model(
        MODEL_PATH,
    )


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
    # Load the model
    model = load_model_from_file()

    # Preprocess the image
    img = preprocess_image(image_path)

    # Get predictions
    predictions = model.predict(img)

    # Extract predictions
    pred_face, pred_age, pred_gender = predictions.values()

    # Process predictions
    age_index = np.argmax(pred_age[0])
    gender_index = np.argmax(pred_gender[0])

    age_text = AGE_GROUPS[age_index]
    gender_text = "Male" if gender_index == 0 else "Female"

    # Retrieve ground truth
    image_name = os.path.basename(image_path)
    gt_info = ground_truth.get(
        f"faces/{image_name}", {"age_group": "Unknown", "gender": "Unknown"}
    )
    gt_age = gt_info["age_group"]
    gt_gender = gt_info["gender"]

    # Print formatted results
    print(f"Image: {image_name}")
    print("Prediction Results:")
    print(f"  Face present probability: {pred_face[0][0]:.2f}")
    print(f"  Predicted Age Group: {age_text}")
    print(f"  Predicted Gender: {gender_text}")
    print(f'  Raw Predictions: {predictions}')
    print("Ground Truth:")
    print(f"  Age Group: {gt_age}")
    print(f"  Gender: {gt_gender}")

    # Load original image using OpenCV
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
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    IMAGES = "./test/faces"
    ground_truth = load_ground_truth()

    for image_name in os.listdir(IMAGES):
        IMAGE_PATH = os.path.join(IMAGES, image_name)

        if not os.path.exists(IMAGE_PATH):
            print(f"Error: Image file '{IMAGE_PATH}' not found")
        else:
            predict_image(IMAGE_PATH, ground_truth)
