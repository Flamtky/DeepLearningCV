import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # True to prevent tensorflow from allocating all GPU memory
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU to use

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
import matplotlib.pyplot as plt
from model import make_siamese_network, make_siamese_network_triplet, compile_siamese_network, euclidean_distance, loss_contrastive
from utils.data_loader import load_data, load_data_triplet, random_seed
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from utils.data_loader import augment_image

batch_size = 32
epochs = 10
input_shape = (224, 224, 3)
margin = 1
data_dir = "./"
data_dir2 = "../Data_Deep-Learning-2024/data/vgg_faces2/"

def start_from_checkpoint(name:str, checkpoint_path: str, init_epoch: int, use_triplet: bool = False):
    # Load Data
    train_dataset, val_dataset = load_dataset(use_triplet)

    loss = loss_triplet if use_triplet else loss_contrastive
    loss_string = "contrastive_loss" if "contrastive" in checkpoint_path else "triplet_loss"

    # Load checkpoint
    siamese_network = load_model(checkpoint_path, custom_objects={"euclidean_distance": euclidean_distance, loss_string: loss})

    siamese_network.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(f"{name}_siamese_best_model.keras", save_best_only=True, monitor="val_loss", mode="min")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, mode="min")
    tensorboard = TensorBoard(log_dir=f"./logs/{name}")

    # Training
    siamese_network.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=init_epoch,
        epochs=epochs,
        callbacks=[checkpoint, reduce_lr, tensorboard],
        verbose=1
    )

    triplet_string = "triplet" if use_triplet else "contrastive"
    siamese_network.save(f"{name}_siamese_final_{triplet_string}_model.h5")

def start_from_new(name:str, use_triplet: bool = False):
    # Load Data
    train_dataset, val_dataset = load_dataset(use_triplet)

    if "finetune" in name:
        print("Finetuning the embedding model")
        weights_path = None
        lr = 1e-3
        fine_tune = 0
        if ".5" in name:
            weights_path = './6.0_embedding_model_triplet.weights.h5'
            lr = 5e-6
            fine_tune = 15
        elif ".6" in name:
            weights_path = './6.5_finetune_embedding_model_triplet.weights.h5'
            lr = 1e-6
            fine_tune = -1
        if use_triplet:
            siamese_network = make_siamese_network_triplet(input_shape, finetune_v5=fine_tune, weights_path=weights_path)
        else:
            siamese_network = make_siamese_network(input_shape, finetune_v5=fine_tune, weights_path=weights_path)
        siamese_network = compile_siamese_network(siamese_network, margin=margin, lr=lr, triplet=use_triplet)
    else:
        if use_triplet:
            if "6.0" in name:
                weights_path = './models/5.3_finetune_embedding_model.weights.h5'
                lr = 1e-5
                fine_tune = 5
                siamese_network = make_siamese_network_triplet(input_shape, finetune_v5=fine_tune, weights_path=weights_path)
                siamese_network = compile_siamese_network(siamese_network, margin=margin, lr=lr, triplet=use_triplet)
            else:
                siamese_network = make_siamese_network_triplet(input_shape)
                siamese_network = compile_siamese_network(siamese_network, margin=margin, triplet=use_triplet)
        else:
            siamese_network = make_siamese_network(input_shape)
            siamese_network = compile_siamese_network(siamese_network, margin=margin, triplet=use_triplet)

    siamese_network.build(input_shape=(batch_size, *input_shape))

    #siamese_network.summary()
    triplet_model = siamese_network.get_layer("siamese_network_triplet")
    triplet_model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(f"{name}_siamese_best_model.keras", save_best_only=True, monitor="val_loss", mode="min")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, mode="min")

    os.makedirs(f"./logs/{name}", exist_ok=True)
    tensorboard = TensorBoard(log_dir=f"./logs/{name}")

    # Training
    handle_fit_with_hard_mining(siamese_network, train_dataset, val_dataset, epochs, [checkpoint, reduce_lr, tensorboard])
    # Save the final model
    triplet_string = "triplet" if use_triplet else "contrastive"
    siamese_network.save(f"{name}_siamese_final_{triplet_string}_model.keras")
    # Save the embedding model weights
    embedding_model = triplet_model.get_layer("embedding_model")
    embedding_model.save_weights(f"{name}_embedding_model_{triplet_string}.weights.h5")

def perform_hard_mining(model, dataset, margin=0.5):
    """
    Perform hard mining to find the hardest positive and negative samples.
    """
    new_anchors = []
    new_positives = []
    new_negatives = []
    for images, paths in tqdm(dataset, desc="Hard Mining", unit="batch", total=len(dataset)):
        anchor, positive, negative = images
        anchor_path, positive_path, negative_path = paths
        ap_distance, an_distance = model.predict([anchor, positive, negative], verbose=0)

        for i, (ap_dis, an_dis) in enumerate(zip(ap_distance, an_distance)):
            if ap_dis > an_dis or an_dis - ap_dis < margin:
                new_anchors.append(anchor_path[i])
                new_positives.append(positive_path[i])
                new_negatives.append(negative_path[i])

    # Build dataset of (anchor, positive, negative)
    dataset = tf.data.Dataset.from_tensor_slices((new_anchors, new_positives, new_negatives))

    # Parse/augment function
    def parse_function(anchor_path, positive_path, negative_path):
        # Anchor
        anchor = tf.io.read_file(anchor_path)
        anchor = tf.image.decode_image(anchor, channels=3)
        anchor.set_shape([None, None, 3])
        anchor = tf.image.resize(anchor, input_shape[:2])
        anchor = augment_image(anchor)
        anchor = anchor / 255.0

        # Positive
        positive = tf.io.read_file(positive_path)
        positive = tf.image.decode_image(positive, channels=3)
        positive.set_shape([None, None, 3])
        positive = tf.image.resize(positive, input_shape[:2])
        positive = augment_image(positive)
        positive = positive / 255.0

        # Negative
        negative = tf.io.read_file(negative_path)
        negative = tf.image.decode_image(negative, channels=3)
        negative.set_shape([None, None, 3])
        negative = tf.image.resize(negative, input_shape[:2])
        negative = augment_image(negative)
        negative = negative / 255.0

        return (anchor, positive, negative), (anchor_path, positive_path, negative_path)

    # Map the parse function
    dataset = dataset.map(
        lambda anchor, positive, negative: parse_function(anchor, positive, negative),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle, batch, prefetch
    dataset = dataset.shuffle(buffer_size=5_000, seed=random_seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def handle_fit_with_hard_mining(model, train_dataset, val_dataset, epochs, callbacks):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Perform hard mining after each epoch
        if epoch in [0, 3]:
            print("Performing hard mining")
            train_dataset = perform_hard_mining(model, train_dataset)

        history = model.fit(
            # lambda to get only the images not the paths; return should be 3 images unpacked fro the tuple
            train_dataset.map(lambda images, paths: images),
            validation_data=val_dataset.map(lambda images, paths: images),
            epochs=epoch + 1,
            initial_epoch=epoch,
            callbacks=callbacks,
            verbose=1
        )


    return history

def load_dataset(use_triplet: bool):
    if use_triplet:
        train_dataset = load_data_triplet(data_dir2, "train", batch_size, input_shape)
        val_dataset = load_data_triplet(data_dir2, "val", batch_size, input_shape)
    else:
        train_dataset = load_data(data_dir, data_dir2, "train", batch_size, input_shape)
        val_dataset = load_data(data_dir, data_dir2, "val", batch_size, input_shape)
    return train_dataset, val_dataset

if __name__ == "__main__":
    start_from_new("6.6_finetune", use_triplet=True)

    """ Saving the embedding model weights
    model = load_model("5.1_finetune_siamese_final_model.keras", custom_objects={"euclidean_distance": euclidean_distance, "contrastive_loss": loss()})
    #save weights of embedding model
    embedding_model = model.get_layer("embedding_model")
    embedding_model.save_weights("5.1_finetune_embedding_model.weights.h5")
    """