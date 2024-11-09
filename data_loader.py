import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import cv2
import numpy as np

AGE_GROUPS = [
    "INFANT",  # 0-4
    "CHILD",  # 5-12
    "TEENAGER",  # 12-21
    "YOUNG_ADULT",  # 21-35
    "ADULT",  # 35-50
    "MIDDLE_AGED",  # 50-65
    "SENIOR",  # 65+
]

age_group_to_index = {group: idx for idx, group in enumerate(AGE_GROUPS)}
gender_to_index = {"M": 0, "F": 1}


def load_data(data_dir, split, batch_size=32, image_size=(224, 224)):
    """
    Load data for a specific split ('train', 'val', 'test').

    Expected folder structure:
    data_dir/
        train/
            faces/
                image.jpg
                ...
            no_faces/
                ...
            faces_anno_train.csv
        val/
            faces/
                ...
            no_faces/
                ...
            faces_anno_val.csv
        test/
            faces/
                ...
            no_faces/
                ...
            faces_anno_test.csv

    The CSV files (faces_anno_{split}.csv) should have the following columns:
    - image_path: Path to the image file relative to the 'faces' directory (e.g., 'faces/image1.jpg')
    - age_group: Age group label
    - gender: Gender label

    'no_faces' directory contains images with no faces and no CSV annotations.
    """

    # Path to the CSV file
    csv_path = os.path.join(data_dir, split, f"faces_anno_{split}.csv")

    # Load face data
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, delimiter=";")
        # Adjust the image paths to be absolute
        df["image_path"] = df["image_path"].apply(
            lambda x: os.path.join(data_dir, split, x)
        )
        df["age_group"] = df["age_group"].map(age_group_to_index)
        df["gender"] = df["gender"].map(gender_to_index)
        df["face"] = 1  # Face present
    else:
        df = pd.DataFrame(columns=["image_path", "age_group", "gender", "face"])

    # Load non-face data
    non_face_dir = os.path.join(data_dir, split, "no_faces")
    if os.path.exists(non_face_dir):
        files = [
            os.path.join(non_face_dir, f)
            for f in os.listdir(non_face_dir)
            if os.path.isfile(os.path.join(non_face_dir, f))
        ]
        non_face_df = pd.DataFrame({"image_path": files})
        non_face_df["age_group"] = 7  # Placeholder
        non_face_df["gender"] = 2  # Placeholder
        non_face_df["face"] = 0
        # Combine face and non-face data
        all_data = pd.concat([df, non_face_df], ignore_index=True)
    else:
        all_data = df

    # Shuffle the data
    all_data = all_data.sample(frac=1).reset_index(drop=True)

    # Create TensorFlow dataset from DataFrame
    image_paths = all_data["image_path"].values
    face_labels = all_data["face"].values.astype(np.float32)
    age_labels = all_data["age_group"].values.astype(np.int32)
    gender_labels = all_data["gender"].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, face_labels, age_labels, gender_labels)
    )

    # Define parsing function
    def parse_function(image_path, face_label, age_label, gender_label):
        # Load and preprocess image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        image = image / 255.0  # Normalize to [0, 1]

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        # Prepare labels
        age_one_hot = tf.one_hot(age_label, depth=len(AGE_GROUPS) + 1)
        gender_label = tf.one_hot(gender_label, depth=2 + 1)

        # Prepare outputs
        age_output = age_one_hot
        gender_output = gender_label

        # Sample weights
        sample_weight_face = 1.0
        sample_weight_age = face_label
        sample_weight_gender = face_label

        return (
            image,
            {"face": face_label, "age": age_output, "gender": gender_output},
            {
                "face": sample_weight_face,
                "age": sample_weight_age,
                "gender": sample_weight_gender,
            },
        )

    # Apply parsing function and batch the dataset
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=5_000, seed=1337)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def display_batch_with_labels(data_dir, batch_size=16, image_size=(224, 224)):
    val_dataset = load_data(
        data_dir, "val", batch_size=batch_size, image_size=image_size
    )
    print(val_dataset.take(1))
    for images, labels, _ in val_dataset.take(1):  # Add '_' to unpack sample weights
        images = images.numpy()
        age_labels = labels["age"].numpy()
        gender_labels = labels["gender"].numpy()
        for i in range(len(images)):
            image = images[i]
            # Denormalize image if necessary
            image = (image * 255).astype(np.uint8)
            age_label = np.argmax(age_labels[i])  # Exclude face_present flag if present
            gender_label = np.argmax(
                gender_labels[i]
            )  # Exclude face_present flag if present
            age_text = (
                AGE_GROUPS[age_label] if age_label < len(AGE_GROUPS) else "Unknown"
            )
            gender_text = (
                "Male"
                if gender_label == 0
                else "Female" if gender_label == 1 else "Unknown"
            )
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Overlay text
            cv2.putText(
                image_bgr,
                f"Age: {age_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image_bgr,
                f"Gender: {gender_text}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            print(f"Age: {age_labels[i]} ({age_label})")
            print(f"Gender: {gender_labels[i]} ({gender_label})")
            # Display the image
            cv2.imshow("Image", image_bgr)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    display_batch_with_labels(data_dir, batch_size=32, image_size=(224, 224))
