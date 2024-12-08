import keras, sys
from keras import layers
import tensorflow as tf

AGE_GROUPS = [
    "INFANT", # 0-4
    "CHILD", # 5-12
    "TEENAGER", # 13-19
    "YOUNG_ADULT", # 20-29
    "ADULT", # 30-49
    "MIDDLE_AGED", # 50-59
    "SENIOR", # 60+
]


def create_model(params, face_only=False):
    use_version: int = params.get("USE_VERSION", 0)

    base_model_class = params["BASE_MODELS"][params["BASE_MODEL_NAME"]]
    base_model = base_model_class(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    base_model.trainable = False
    # Unfreezing the last 5 layers of the base model
    for layer in base_model.layers[:-5]:
        layer.trainable = True

    inputs = keras.Input(shape=(224, 224, 3))
    base = base_model(inputs, training=False)
    base = layers.GlobalAveragePooling2D()(base)

    outputs = {}
    if face_only:
        outputs = createSingelTaskModel(params, base, outputs)
    else:
        if use_version == 0:
            outputs = createMultiTaskModel_old(params, base, outputs)
        elif use_version == 1:
            outputs = createMultiTaskModel(params, base, outputs)
        elif use_version == 2:
            outputs = createMultiTaskModel_v2(params, base, outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def createSingelTaskModel(params, base_model, outputs):
    singelTask_model = base_model
    for _ in range(params["NUMBER_OF_DENSE_LAYERS"]):
        singelTask_model = layers.Dense(
            params["DENSE_LAYER_SIZE"],
            activation=params["DENSE_ACTIVATION_FUNCTION"],
            kernel_regularizer=keras.regularizers.l2(params["WEIGHT_DECAY"]),
        )(singelTask_model)
        if params["USE_BATCH_NORM"]:
            singelTask_model = layers.BatchNormalization()(singelTask_model)
        if params["DROPOUT_RATE"] > 0:
            singelTask_model = layers.Dropout(params["DROPOUT_RATE"])(singelTask_model)
    singelTask_model = layers.Dense(1, activation="sigmoid", name="face")(singelTask_model)
    outputs["face"] = singelTask_model

    return outputs

def createMultiTaskModel_old(params, base_model, outputs):
    for output_name in ["face", "age", "gender"]:
        multiTask_model = base_model
        for _ in range(params["NUMBER_OF_DENSE_LAYERS"]):
            multiTask_model = layers.Dense(
                    params["DENSE_LAYER_SIZE"],
                    activation=params["DENSE_ACTIVATION_FUNCTION"],
                    kernel_regularizer=keras.regularizers.l2(params["WEIGHT_DECAY"]),
                )(multiTask_model)
            if params["USE_BATCH_NORM"]:
                multiTask_model = layers.BatchNormalization()(multiTask_model)
            if params["DROPOUT_RATE"] > 0:
                multiTask_model = layers.Dropout(params["DROPOUT_RATE"])(multiTask_model)
        if output_name == "face":
            multiTask_model = layers.Dense(1, activation="sigmoid", name="face")(multiTask_model)
        elif output_name == "age":
            multiTask_model = layers.Dense(len(AGE_GROUPS), activation="softmax", name="age")(multiTask_model)
        elif output_name == "gender":
            multiTask_model = layers.Dense(2, activation="softmax", name="gender")(multiTask_model)
        outputs[output_name] = multiTask_model

    return outputs

def createMultiTaskModel(params, base_model, outputs):
    # Dense
    # BatchNormalization
    # Activation
    # Dropout
    mutliTask_model = base_model

    regularizer = None
    if params["WEIGHT_DECAY"] > 0:
        regularizer = keras.regularizers.l2(params["WEIGHT_DECAY"])

    face = layers.Dense(128, activation=None, kernel_regularizer=regularizer)(mutliTask_model)
    face = _generateBNandDropoutLayer(face, params["DENSE_ACTIVATION_FUNCTION"], params["DROPOUT_RATE"], params["USE_BATCH_NORM"])
    face = layers.Dense(1, activation="sigmoid", name="face")(face)

    age = layers.Dense(512, activation=None, kernel_regularizer=regularizer)(mutliTask_model)
    age = _generateBNandDropoutLayer(age, params["DENSE_ACTIVATION_FUNCTION"], params["DROPOUT_RATE"], params["USE_BATCH_NORM"])
    age = layers.Dense(512, activation=None, kernel_regularizer=regularizer)(age)
    age = _generateBNandDropoutLayer(age, params["DENSE_ACTIVATION_FUNCTION"], params["DROPOUT_RATE"], params["USE_BATCH_NORM"])
    age = layers.Dense(512, activation=None, kernel_regularizer=regularizer)(age)
    age = _generateBNandDropoutLayer(age, params["DENSE_ACTIVATION_FUNCTION"], params["DROPOUT_RATE"], params["USE_BATCH_NORM"])
    age = layers.Dense(len(AGE_GROUPS), activation="softmax", name="age")(age)

    gender = layers.Dense(512, activation=None, kernel_regularizer=regularizer)(mutliTask_model)
    gender = _generateBNandDropoutLayer(gender, params["DENSE_ACTIVATION_FUNCTION"], params["DROPOUT_RATE"], params["USE_BATCH_NORM"])
    gender = layers.Dense(2, activation="softmax", name="gender")(gender)

    outputs["face"] = face
    outputs["age"] = age
    outputs["gender"] = gender

    return outputs

def createMultiTaskModel_v2(_params, base_model, outputs):
    # Dense
    # BatchNormalization
    # Activation
    # Dropout
    mutliTask_model = base_model

    face = layers.Dense(128, activation=None)(mutliTask_model)
    face = _generateBNandDropoutLayer(face, "leaky_relu", 0.0, False)
    face = layers.Dense(1, activation="sigmoid", name="face")(face)

    age = layers.Dense(512, activation=None)(mutliTask_model)
    age = _generateBNandDropoutLayer(age, "leaky_relu", 0.2, False)
    age = layers.Dense(2, activation="softmax", name="age")(age) # only isYoung

    gender = layers.Dense(512, activation=None)(mutliTask_model)
    gender = _generateBNandDropoutLayer(gender, "leaky_relu", 0.2, False)
    gender = layers.Dense(2, activation="softmax", name="gender")(gender)

    outputs["face"] = face
    outputs["age"] = age
    outputs["gender"] = gender

    return outputs

def _generateBNandDropoutLayer(model, activation, dropout_rate, use_batch_norm):
    if use_batch_norm:
        model = layers.BatchNormalization()(model)
    model = layers.Activation(activation)(model)
    if dropout_rate > 0:
        model = layers.Dropout(dropout_rate)(model)
    return model

@keras.saving.register_keras_serializable()
def conditional_age_loss(y_true, y_pred):
    loss = keras.losses.binary_crossentropy(y_true[:, 0:2], y_pred)
    face_present = tf.cast(y_true[:, -1], tf.float32)

    loss = tf.where(face_present <= 0.9, loss, 0.0)
    if tf.executing_eagerly():
        tf.print("---------conditional_age_loss-----------")
        tf.print("face_present: ", face_present.numpy())
        tf.print("y_true: ", y_true.numpy())
        tf.print("y_pred: ", y_pred.numpy())
        tf.print("loss: ", loss.numpy())
        tf.print("--------------------")
    return tf.reduce_mean(loss)


@keras.saving.register_keras_serializable()
def conditional_gender_loss(y_true, y_pred):
    loss = keras.losses.binary_crossentropy(y_true[:, 0:2], y_pred)
    face_present = tf.cast(y_true[:, -1], tf.float32)

    loss = tf.where(face_present <= 0.9, loss, 0.0)
    if tf.executing_eagerly():
        tf.print("---------conditional_gender_loss-----------")
        tf.print("face_present: ", face_present.numpy())
        tf.print("y_true: ", y_true.numpy())
        tf.print("y_pred: ", y_pred.numpy())
        tf.print("loss: ", loss.numpy())
        tf.print("--------------------")
    return tf.reduce_mean(loss)


@keras.saving.register_keras_serializable()
def face_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred)


@keras.saving.register_keras_serializable()
def age_acc(y_true, y_pred):
    face_present = tf.cast(y_true[:, -1] <= 0.9, tf.float32)
    true_labels = tf.argmax(y_true, axis=-1)
    pred_labels = tf.argmax(y_pred, axis=-1)
    correct = tf.cast(tf.equal(true_labels, pred_labels), tf.float32) * face_present
    accuracy = tf.reduce_sum(correct) / (
        tf.reduce_sum(face_present) + tf.keras.backend.epsilon()
    )
    if tf.executing_eagerly():
        tf.print("---------age_acc-----------")
        tf.print("face_present: ", face_present.numpy())
        tf.print("true_labels: ", true_labels.numpy())
        tf.print("pred_labels: ", pred_labels.numpy())
        tf.print("correct: ", correct.numpy())
        tf.print("accuracy: ", accuracy.numpy())
        tf.print("--------------------")
    return accuracy


@keras.saving.register_keras_serializable()
def gen_acc(y_true, y_pred):
    face_present = tf.cast(y_true[:, -1] <= 0.9, tf.float32)
    true_labels = tf.argmax(y_true, axis=-1)
    pred_labels = tf.argmax(y_pred, axis=-1)
    correct = tf.cast(tf.equal(true_labels, pred_labels), tf.float32) * face_present
    accuracy = tf.reduce_sum(correct) / (
        tf.reduce_sum(face_present) + tf.keras.backend.epsilon()
    )
    if tf.executing_eagerly():
        tf.print("---------gen_acc-----------")
        tf.print("face_present: ", face_present.numpy())
        tf.print("true_labels: ", true_labels.numpy())
        tf.print("pred_labels: ", pred_labels.numpy())
        tf.print("correct: ", correct.numpy())
        tf.print("accuracy: ", accuracy.numpy())
        tf.print("--------------------")
    return accuracy
