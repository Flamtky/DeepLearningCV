import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import cv2
import numpy as np

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random

random_seed = 1337
MAX_IDENTITY = 500
USE_PERCENTAGE = 0.7
ITER_PER_ANCHOR = 2
image_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal", seed=random_seed), # Mirror
        keras.layers.RandomRotation(0.2, interpolation="nearest", seed=random_seed), # 20*2pi
        keras.layers.RandomZoom(0.2, seed=random_seed),
        keras.layers.RandomContrast(0.2, seed=random_seed),
        keras.layers.RandomBrightness(0.1, seed=random_seed),
    ]
)

def load_data(
    data_dir,
    data_dir2,
    split,
    batch_size=32,
    image_size=(224, 224, 3),
    seed=random_seed
):
    """
    Load data for a specific split ('train', 'val', 'test') and return a
    tf.data.Dataset of pairs (image1, image2) and labels (1 if same identity, 0 otherwise).

    Arguments:
        data_dir (str): Root data directory. Lagenda
        data_dir2 (str): Root data directory. VGGFACE2
        split (str): One of 'train', 'val', or 'test'.
        batch_size (int): Batch size for the tf.data.Dataset.
        image_size (tuple): Desired (width, height, channels) for the images.
        seed (int): Random seed for shuffling, etc.

    Returns:
        tf.data.Dataset: A dataset yielding ((image1, image2), label).
    """

    # Load the CSV containing image paths for faces
    csv_path = os.path.join(data_dir, split, f"faces_anno_{split}.csv")
    if not os.path.exists(csv_path):
        # If the CSV doesn't exist, return an empty dataset or raise an error
        print(f"Warning: CSV for {split} not found at {csv_path}. Returning empty dataset.")
        return tf.data.Dataset.from_tensors(([], []))

    df = pd.read_csv(csv_path, delimiter=";")

    # Adjust the image paths to be absolute (and replace .jpg with .png if needed)
    df["image_path"] = df["image_path"].apply(
        lambda x: os.path.join(data_dir, split, x.replace(".jpg", ".png"))
    )

    lagenda_image_paths = df["image_path"].values

    ## VGGFACE2 dataset
    vgg_split = split
    if split == "val": # test is the validation set for vggface2
        vgg_split = "test"
        print("Warning: VGGFACE2 test set is used as validation set.")
    vgg_path = os.path.join(data_dir2, vgg_split)
    # dict key = identity, value = list of image paths
    vgg_all_ids: dict[str, list[str]] = {}
    for i, f_id in enumerate(os.listdir(vgg_path)):
        if MAX_IDENTITY != -1 and i >= MAX_IDENTITY:
            break
        if os.path.isdir(os.path.join(vgg_path, f_id)):
            vgg_all_ids[f_id] = []
            for img in os.listdir(os.path.join(vgg_path, f_id)):
                vgg_all_ids[f_id].append(os.path.join(vgg_path, f_id, img))

    # Generate pairs
    positive_pairs = []
    negative_pairs = []

    # Set a random seed for reproducibility
    random.seed(seed)

    ## Lagenda ##
    # Positive pairs: same identity (since each image is a unique identity,
    # we'll pair it with itself).
    #for img_path in lagenda_image_paths:
    #    positive_pairs.append((img_path, img_path, 2)) # Will be later converted to 0; Special case for agumentation

    # Negative pairs: different identity. We pick a random different image_path
    # from the dataset.
    #for img_path in lagenda_image_paths:
    #    # Randomly pick a different image
    #    neg_img_path = random.choice(lagenda_image_paths)
    #    # Ensure it's different
    #    while neg_img_path == img_path:
    #        neg_img_path = random.choice(lagenda_image_paths)
    #negative_pairs.append((img_path, neg_img_path, 1))

    ## VGGFACE2 ##
    # Positive pairs: same identity
    # Generate positive pairs for VGGFACE2: pair each image with every other image of the same identity.
    # Example: For images [img1, img2, img3], generate pairs (img1, img2), (img1, img3), and (img2, img3).
    vgg_all_ids_keys = list(vgg_all_ids.keys())
    for identity, img_paths in vgg_all_ids.items():
        for i in range(len(img_paths)):
            for j in range(i + 1, len(img_paths)):
                positive_pairs.append((img_paths[i], img_paths[j], 0))

    # Negative pairs: different identity
    all_ids = list(vgg_all_ids.keys())
    for identity, img_paths in vgg_all_ids.items():
        for i in range(len(img_paths)):
            random_id2 = random.choice(vgg_all_ids_keys)
            while random_id2 == identity:
                random_id2 = random.choice(all_ids)
            random_img = random.choice(vgg_all_ids[random_id2])
            negative_pairs.append((img_paths[i], random_img, 1))

    # Balance the number of positive and negative pairs
    old_positive_pairs = len(positive_pairs)
    old_negative_pairs = len(negative_pairs)
    max_pairs = min(old_positive_pairs, old_negative_pairs)

    # Shuffle and truncate
    random.shuffle(positive_pairs) # shuffle in-place
    random.shuffle(negative_pairs) # shuffle in-place
    positive_pairs = positive_pairs[:max_pairs]
    negative_pairs = negative_pairs[:max_pairs]
    print(f"Balanced pairs: Positive: {old_positive_pairs} -> {len(positive_pairs)}, Negative: {old_negative_pairs} -> {len(negative_pairs)}")

    # Combine
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)  # Shuffle again

    # Cutoff
    all_pairs = all_pairs[:int(len(all_pairs) * USE_PERCENTAGE)]

    # Convert to a DataFrame so we can more easily vectorize in tf.data
    all_pairs_df = pd.DataFrame(all_pairs, columns=["image_path_1", "image_path_2", "label"])

    # Create slices of the columns
    ds_image_path_1 = tf.constant(all_pairs_df["image_path_1"].values, dtype=tf.string)
    ds_image_path_2 = tf.constant(all_pairs_df["image_path_2"].values, dtype=tf.string)
    ds_label        = tf.constant(all_pairs_df["label"].values, dtype=tf.int32)

    # Build dataset of ((path1, path2), label)
    dataset = tf.data.Dataset.from_tensor_slices(((ds_image_path_1, ds_image_path_2), ds_label))

    # Parse/augment function
    def parse_function(image_path_1, image_path_2, label):
        # Image 1
        image1 = tf.io.read_file(image_path_1)
        image1 = tf.image.decode_image(image1, channels=3)
        image1.set_shape([None, None, 3])
        image1 = tf.image.resize(image1, image_size[:2])

        # Augmentation (only if no lagenda postive pair, as we want to keep one original image)
        if label != 2:
            image1 = augment_image(image1)
        else:
            label = 0
        image1 = image1 / 255.0  # normalize to [0, 1]

        # Image 2
        image2 = tf.io.read_file(image_path_2)
        image2 = tf.image.decode_image(image2, channels=3)
        image2.set_shape([None, None, 3])
        image2 = tf.image.resize(image2, image_size[:2])

        image2 = augment_image(image2)
        image2 = image2 / 255.0  # normalize to [0, 1]

        return (image1, image2), label

    # Map the parse function
    dataset = dataset.map(
        lambda paths, lbl: parse_function(paths[0], paths[1], lbl),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle, batch, prefetch
    dataset = dataset.shuffle(buffer_size=1_000, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def load_data_triplet(
    data_dir,
    split,
    batch_size=32,
    image_size=(224, 224, 3),
    seed=random_seed
):
    """
    Load data for a specific split ('train', 'val', 'test') and return a
    tf.data.Dataset of triplets (anchor, positive, negative).

    Arguments:
        data_dir (str): Root data directory. VGGFACE2
        split (str): One of 'train', 'val', or 'test'.
        batch_size (int): Batch size for the tf.data.Dataset.
        image_size (tuple): Desired (width, height, channels) for the images.
        seed (int): Random seed for shuffling, etc.

    Returns:
        tf.data.Dataset: A dataset yielding (anchor, positive, negative).
    """

    ## VGGFACE2 dataset
    vgg_split = split
    if split == "val": # test is the validation set for vggface2
        vgg_split = "test"
        print("Warning: VGGFACE2 test set is used as validation set.")
    vgg_path = os.path.join(data_dir, vgg_split)
    # dict key = identity, value = list of image paths
    vgg_all_ids: dict[str, list[str]] = {}
    for i, f_id in enumerate(os.listdir(vgg_path)):
        if MAX_IDENTITY != -1 and i >= MAX_IDENTITY:
            break
        if os.path.isdir(os.path.join(vgg_path, f_id)):
            vgg_all_ids[f_id] = []
            for img in os.listdir(os.path.join(vgg_path, f_id)):
                vgg_all_ids[f_id].append(os.path.join(vgg_path, f_id, img))

    # Generate triplets
    triplets = []

    # Set a random seed for reproducibility
    random.seed(seed)

    all_ids = list(vgg_all_ids.keys())
    for identity, img_paths in vgg_all_ids.items():
        for i in range(len(img_paths)):
            anchor = img_paths[i]
            # Sample ITER_PER_ANCHOR random positives
            positives = random.sample(img_paths[:i] + img_paths[i+1:], min(ITER_PER_ANCHOR, len(img_paths) - 1))
            for positive in positives:
                random_id = random.choice(all_ids)
                while random_id == identity:
                    random_id = random.choice(all_ids)
                negative = random.choice(vgg_all_ids[random_id])
                triplets.append((anchor, positive, negative))

    # Shuffle
    random.shuffle(triplets)
    # Cutoff
    triplets = triplets[:int(len(triplets) * USE_PERCENTAGE)]

    print(f"Triplets: {len(triplets)}")

    # Convert to a DataFrame so we can more easily vectorize in tf.data
    triplets_df = pd.DataFrame(triplets, columns=["anchor", "positive", "negative"])

    # Create slices of the columns
    ds_anchor = tf.constant(triplets_df["anchor"].values, dtype=tf.string)
    ds_positive = tf.constant(triplets_df["positive"].values, dtype=tf.string)
    ds_negative = tf.constant(triplets_df["negative"].values, dtype=tf.string)

    # Build dataset of (anchor, positive, negative)
    dataset = tf.data.Dataset.from_tensor_slices((ds_anchor, ds_positive, ds_negative))

    # Parse/augment function
    def parse_function(anchor_path, positive_path, negative_path):
        # Anchor
        anchor = tf.io.read_file(anchor_path)
        anchor = tf.image.decode_image(anchor, channels=3)
        anchor.set_shape([None, None, 3])
        anchor = tf.image.resize(anchor, image_size[:2])
        anchor = augment_image(anchor)
        anchor = anchor / 255.0

        # Positive
        positive = tf.io.read_file(positive_path)
        positive = tf.image.decode_image(positive, channels=3)
        positive.set_shape([None, None, 3])
        positive = tf.image.resize(positive, image_size[:2])
        positive = augment_image(positive)
        positive = positive / 255.0

        # Negative
        negative = tf.io.read_file(negative_path)
        negative = tf.image.decode_image(negative, channels=3)
        negative.set_shape([None, None, 3])
        negative = tf.image.resize(negative, image_size[:2])
        negative = augment_image(negative)
        negative = negative / 255.0

        return (anchor, positive, negative), (anchor_path, positive_path, negative_path)

    # Map the parse function
    dataset = dataset.map(
        lambda anchor, positive, negative: parse_function(anchor, positive, negative),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle, batch, prefetch
    dataset = dataset.shuffle(buffer_size=5_000, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def augment_image(image):
    i = image_augmentation(image)
    return i

def display_batch_for_debugging(data_dir=".", data_dir2="../Data_Deep-Learning-2024/data/vgg_faces2/"):
    """
    Display a batch of images for debugging purposes.
    """
    import matplotlib.pyplot as plt

    batch_size = 4

    dataset = load_data(data_dir, data_dir2, "train", batch_size=batch_size)

    for (image1, image2), labels in dataset.take(1):
        _, axs = plt.subplots(2, 2, figsize=(10, 10))
        for i in range(batch_size):
            ax = axs[i // 2, i % 2]
            concat_image = np.concatenate([image1[i].numpy(), image2[i].numpy()], axis=1)
            ax.imshow(concat_image)
            ax.set_title(f"Label: {labels[i].numpy()}")
            ax.axis("off")

        plt.savefig("batch.png")

def display_triplet_batch_for_debugging(data_dir="../Data_Deep-Learning-2024/data/vgg_faces2/"):
    """
    Display a batch of triplets for debugging purposes.
    """
    import matplotlib.pyplot as plt

    batch_size = 16

    dataset = load_data_triplet(data_dir, "train", batch_size=batch_size)

    for anchor, positive, negative in dataset.take(1):
        _, axs = plt.subplots(3, batch_size, figsize=(16, 8))
        for i in range(batch_size):
            ax = axs[0, i]
            ax.imshow(anchor[i].numpy())
            ax.set_title("Anchor")
            ax.axis("off")

            ax = axs[1, i]
            ax.imshow(positive[i].numpy())
            ax.set_title("Positive")
            ax.axis("off")

            ax = axs[2, i]
            ax.imshow(negative[i].numpy())
            ax.set_title("Negative")
            ax.axis("off")

        plt.savefig("triplet_batch.png")

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
    data_dir = "../Data_Deep-Learning-2024/data/vgg_faces2/"
    #display_batch_for_debugging(data_dir)
    display_triplet_batch_for_debugging(data_dir)
