import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from model import make_siamese_network_triplet, compile_siamese_network

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = (
    "./6.0_embedding_model_triplet.weights.h5"
)

def load_and_preprocess_image(image_path, image_size=(224, 224)):
    """Load and preprocess an image."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image

def predict_similarity(model, img1_path, img2_path, img3_path):
    """Predict similarity between two images."""
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    img3 = load_and_preprocess_image(img3_path)

    img1 = tf.expand_dims(img1, axis=0)
    img2 = tf.expand_dims(img2, axis=0) 
    img3 = tf.expand_dims(img3, axis=0)

    #['input_anchor', 'input_positive', 'input_negative']
    prediction = model.predict({"input_anchor": img1, "input_positive": img2, "input_negative": img3})
    return prediction

def plot_results(img1_path, img2_path, img3_path, prediction):
    """Plot the two images with the similarity score."""
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    img3 = load_and_preprocess_image(img3_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img1)
    axes[0].axis("off")
    axes[0].set_title(img1_path)

    axes[1].imshow(img2)
    axes[1].axis("off")
    axes[1].set_title(img2_path)

    axes[2].imshow(img3)
    axes[2].axis("off")
    axes[2].set_title(img3_path)

    similarity = str(prediction)

    fig.suptitle(f"Similarity: {similarity}", fontsize=16, y=0.1, color="black", va="center", ha="center", backgroundcolor="white")
    plt.tight_layout()
    plt.savefig("result.png")
    #plt.show()

def get_image_path(data_dir, index):
    """Get the image path by walking over the data_dir until the specified index is reached."""
    if isinstance(index, int):
        for root, _, files in os.walk(data_dir):
            files = sorted(files)
            if index < len(files):
                return os.path.join(root, files[index])
            index -= len(files)
        raise IndexError("Index out of range")
    elif isinstance(index, str):
        # given is relative to my path not to data
        return index
    else:
        raise ValueError("Index must be an integer or a string representing the image path")

if __name__ == "__main__":
    # Load model
    input_shape = (224, 224, 3)
    siam_model = make_siamese_network_triplet((224, 224, 3), -1, MODEL_PATH)
    model = compile_siamese_network(siam_model, triplet=True)

    data_dir = "."
    img1_index = "1.png"
    img2_index = "2.png"
    img3_index = "4.png"

    # Load images
    img1_path = get_image_path(data_dir, img1_index)
    img2_path = get_image_path(data_dir, img2_index)
    img3_path = get_image_path(data_dir, img3_index)

    # Predict similarity
    similarity = predict_similarity(model, img1_path, img2_path, img3_path)

    # Plot results
    plot_results(img1_path, img2_path, img3_path, similarity)
