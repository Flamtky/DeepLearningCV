from keras import layers, Model, optimizers, metrics
from keras import ops
import keras
import tensorflow as tf
from keras.applications import ResNet50

@keras.saving.register_keras_serializable()
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, keras.backend.epsilon()))

class L2NormalizeLayer(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(L2NormalizeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return keras.utils.normalize(inputs, axis=self.axis)

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = ops.sum(tf.square(anchor - positive), -1)
        an_distance = ops.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class SiameseModelTriplet(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.accuracy = metrics.Mean(name="accuracy")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            acc, loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(acc)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy.result()}

    def test_step(self, data):
        acc, loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(acc)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)

        # Accuracy
        acc = tf.reduce_mean(tf.cast(ap_distance < an_distance, tf.float32), axis=0)
        return acc, loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.accuracy]

def make_embedding_modelv1(input_shape):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    inputs = layers.Input(input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512)(x) 
    return Model(inputs, x)

def make_embedding_modelv2(input_shape):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    inputs = layers.Input(input_shape)
    x = base_model(inputs, training=False)
    x = layers.Flatten()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)

    x = L2NormalizeLayer()(x)
    return Model(inputs, x, name="embedding_model")

def make_embedding_modelv3(input_shape):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze base model, but the last layers are trainable
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-15:]:
        layer.trainable = True

    inputs = layers.Input(input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = L2NormalizeLayer()(x)
    return Model(inputs, x, name="embedding_model")

def make_embedding_modelv4(input_shape):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze base model, but the last layers are trainable
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-5:]:
        layer.trainable = True

    inputs = layers.Input(input_shape)
    x = base_model(inputs, training=False)
    x = layers.Flatten()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = L2NormalizeLayer()(x)
    return Model(inputs, x, name="embedding_model")

def make_embedding_modelv5(input_shape):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    base_model.trainable = True

    inputs = layers.Input(input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    
    x = L2NormalizeLayer()(x)
    return Model(inputs, x, name="embedding_model")

def make_embedding_modelv5_finetune(input_shape, base_model_trainable_layers=5, weights_path='embedding_modelv5.h5'):
    # Load the pre-trained model
    model = make_embedding_modelv5(input_shape)
    if weights_path is None:
        raise ValueError("weights_path cannot be None (given: None)")
    model.load_weights(weights_path)

    # Freeze the base model
    base_model = model.layers[1]
    if base_model_trainable_layers == -1:
        print("Unfreezing all layers")
        # Unfreeze all layers
        base_model.trainable = True
    else:
        print(f"Unfreezing the last {base_model_trainable_layers} layers")
        base_model.trainable = False
        # Unfreeze the last `base_model_trainable_layers` layers
        for layer in base_model.layers[-base_model_trainable_layers:]:
            layer.trainable = True

    return model

def make_siamese_network(input_shape, finetune_v5:int=0, weights_path=None):
    if finetune_v5 != 0:
        embedding_model = make_embedding_modelv5_finetune(input_shape, finetune_v5, weights_path)
    else:
        embedding_model = make_embedding_modelv5(input_shape)

    input_1 = layers.Input(shape=input_shape, name="input_1")
    input_2 = layers.Input(shape=input_shape, name="input_2")

    tower_1 = embedding_model(input_1)
    tower_2 = embedding_model(input_2)

    merge_layer = layers.Lambda(euclidean_distance, output_shape=(1,))([tower_1, tower_2])
    normal_layer = layers.BatchNormalization()(merge_layer)
    outputs = layers.Dense(1, activation="sigmoid")(normal_layer)

    siamese_network = Model(inputs=[input_1, input_2], outputs=outputs, name="siamese_network_contrastive")
    return siamese_network

def make_siamese_network_triplet(input_shape, finetune_v5=0, weights_path=None):
    if finetune_v5 != 0:
        embedding_model = make_embedding_modelv5_finetune(input_shape, finetune_v5, weights_path)
    else:
        embedding_model = make_embedding_modelv5(input_shape)

    input_anchor = layers.Input(shape=input_shape, name="input_anchor")
    input_positive = layers.Input(shape=input_shape, name="input_positive")
    input_negative = layers.Input(shape=input_shape, name="input_negative")

    tower_anchor = embedding_model(input_anchor)
    tower_positive = embedding_model(input_positive)
    tower_negative = embedding_model(input_negative)

    outputs = DistanceLayer()(tower_anchor, tower_positive, tower_negative)

    model = Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=outputs,
        name="siamese_network_triplet"
    )
    siamese_network = SiameseModelTriplet(model, margin=0.5)
    return siamese_network

def loss_contrastive(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    @keras.saving.register_keras_serializable()
    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """
        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss

def compile_siamese_network(model, margin=1, lr=1e-3, triplet=False):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=None if triplet else loss_contrastive(margin),
        metrics=None if triplet else ["accuracy"]
    )
    return model

if __name__ == "__main__":
    input_shape = (224, 224, 3)
    margin = 1

    siamese_network = make_siamese_network(input_shape)
    siamese_network = compile_siamese_network(siamese_network, margin=margin)

    siamese_network.summary()
