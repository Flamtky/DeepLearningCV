import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

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
    Load face image data for a specific split ('train', 'val', 'test').
    Expected structure:
    data_dir/
        train/
            images/
                image.jpg
                ...
            anno_train.csv
        val/
            images/
                ...
            anno_val.csv
    """
    # Path to the CSV file
    csv_path = os.path.join(data_dir, split, f"faces_anno_{split}.csv")
    
    # Load data
    df = pd.read_csv(csv_path, delimiter=";")
    df["image_path"] = df["image_path"].apply(
        lambda x: os.path.join(data_dir, split, x)
    )
    df["age_group"] = df["age_group"].map(age_group_to_index)
    df["gender"] = df["gender"].map(gender_to_index)

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Create TensorFlow dataset
    image_paths = df["image_path"].values
    age_labels = df["age_group"].values.astype(np.int32)
    gender_labels = df["gender"].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, age_labels, gender_labels)
    )

    def parse_function(image_path, age_label, gender_label):
        # Load and preprocess image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        image = image / 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]

        # Data augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.clip_by_value(image, -1.0, 1.0)

        # Prepare one-hot labels
        age_one_hot = tf.one_hot(age_label, depth=len(AGE_GROUPS))
        gender_one_hot = tf.one_hot(gender_label, depth=2)

        return image, {"age": age_one_hot, "gender": gender_one_hot}

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
    for images, labels in val_dataset.take(1):
        images = images.numpy()
        age_labels = labels["age"].numpy()
        gender_labels = labels["gender"].numpy()
        for i in range(len(images)):
            image = images[i]
            # Denormalize image if necessary
            image = ((image + 1.0) * 127.5).astype(np.uint8)
            age_label = np.argmax(age_labels[i])
            gender_label = np.argmax(gender_labels[i])
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
