import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # True to prevent tensorflow from allocating all GPU memory
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU to use

WEIGHTS_PATH = '6.6_finetune_embedding_model_triplet.weights.h5'
EMBEDDING_DB_PATH = './embedding_db.npy'
DATA_DIR = "../Data_Deep-Learning-2024/data/vgg_faces2/test"
import numpy as np
from tqdm import tqdm

def create_embedding_model():
    """
    Creates and returns an embedding model using the make_embedding_modelv5_finetune function.
    This function imports the make_embedding_modelv5_finetune function from the model module,
    and uses it to create an embedding model with the specified input shape, number of classes,
    and weights path.
    Returns:
        model: The created embedding model.
    """
    from model import make_embedding_modelv5_finetune
    model = make_embedding_modelv5_finetune((224, 224, 3), -1, WEIGHTS_PATH)

    return model

def get_data() -> dict:
    """
    Retrieves a dictionary containing lists of file paths for each directory in the specified data directory.
    Returns:
        dict: A dictionary where the keys are directory names and the values are lists of file paths for each file in the corresponding directory.
    """
    data_all_ids: dict[str, list[str]] = {}
    for i, f_id in enumerate(os.listdir(DATA_DIR)):
        if os.path.isdir(os.path.join(DATA_DIR, f_id)):
            data_all_ids[f_id] = []
            for img in os.listdir(os.path.join(DATA_DIR, f_id)):
                data_all_ids[f_id].append(os.path.join(DATA_DIR, f_id, img))

    return data_all_ids

def load_img(path):
    """
    Loads and preprocesses an image from the given file path.

    Args:
        path (str): The file path to the image.

    Returns:
        tf.Tensor: A tensor representing the preprocessed image with shape (224, 224, 3).
    """
    import tensorflow as tf
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img

def save_embedding():
    """
    Generates and saves embeddings for images using a pre-trained embedding model.
    This function performs the following steps:
    1. Loads a pre-trained embedding model.
    2. Retrieves image paths grouped by identity.
    3. For each identity, loads the images, generates embeddings using the model, 
       and stores the embeddings along with their corresponding image paths.
    4. Saves the embeddings dictionary to a file.
    The embeddings dictionary has the following structure:
    {
        identity: [
            [embedding1, embedding2, ...], 
            [embedding_img_path1, embedding_img_path2, ...]
        ],
        ...
    }
    """
    import tensorflow as tf

    model = create_embedding_model()
    img_paths_per_id = get_data()

    embeddings = {} # {identity: [[embedding1, embedding2, ...], [embedding_img_path1, embedding_img_path2, ...]]}
    for identity, img_paths in tqdm(img_paths_per_id.items()):
        embeddings[identity] = []
        images = [load_img(img_path) for img_path in img_paths]
        images = tf.stack(images)
        embedding = model.predict(images, verbose=0)
        combined = [embedding, img_paths]
        embeddings[identity].append(combined)

    np.save(EMBEDDING_DB_PATH, embeddings)
    

def load_embedding():
    return np.load(EMBEDDING_DB_PATH, allow_pickle=True).item()

def visualize_embedding_as_cluster():
    """
    Visualizes embeddings as a cluster using t-SNE and matplotlib.
    This function loads embeddings, reduces their dimensionality using t-SNE, 
    and plots them in a 2D space. Each identity is represented by a different color, 
    and the corresponding images are displayed at their respective positions.
    Steps:
    1. Load embeddings.
    2. Limit the number of images per identity to MAX_IMAGES_PER_IDENTITY.
    3. Reduce dimensionality of embeddings using t-SNE.
    4. Plot the 2D embeddings with different colors for each identity.
    5. Overlay the corresponding images on the plot.

    Saves:
    A plot named 'top10_embedding_cluster.png' in the current directory.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    embeddings = load_embedding()
    for identity in embeddings.keys():
        print(f'Found {len(embeddings[identity][0][0])} embeddings for {identity}')

    X = []
    y = []
    img_paths = []
    MAX_IMAGES_PER_IDENTITY = 10

    for identity, emb_list in embeddings.items():
        embedding_list = emb_list[0][0][:MAX_IMAGES_PER_IDENTITY]
        img_paths.extend(emb_list[0][1][:MAX_IMAGES_PER_IDENTITY])
        X.extend(embedding_list)
        y.extend([identity] * len(embedding_list))

    X = np.array(X)
    y = np.array(y)

    tsne = TSNE(n_components=2, random_state=42, perplexity=10, learning_rate=36.5)
    X_2d = tsne.fit_transform(X)

    target_ids = range(len(embeddings.keys()))
    plt.figure(figsize=(20*10, 16*10))

    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, embeddings.keys()):
        plt.scatter(X_2d[y == label, 0], X_2d[y == label, 1], c=c, label=label, s=10)

    for i in tqdm(range(len(X_2d)), desc="Adding images", unit="image"):
        img = plt.imread(img_paths[i])
        imagebox = OffsetImage(img)
        ab = AnnotationBbox(imagebox, (X_2d[i, 0], X_2d[i, 1]), frameon=False)
        plt.gca().add_artist(ab)

    plt.legend()
    plt.savefig('top10_embedding_cluster.png')

if __name__ == '__main__':
    #save_embedding()
    visualize_embedding_as_cluster()
