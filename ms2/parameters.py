import logging
from keras import applications, activations, optimizers

logging.basicConfig(
    level=logging.WARNING, format="%(levelname)s: %(message)s", filename="error.log"
)

FACE_ONLY = False
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LOSS_WEIGHTS = {
    "face": 0.5,
    "age": 1.0,
    "gender": 1.5,
}

LR = {
    "initials": [
        1e-5,
        1e-4,
        1e-3,
    ],
    "decays": [
        0.9,
        0.95,
    ],
}

BASE_MODELS = {
    "efficientnetb0": applications.EfficientNetB0,
    "efficientnetb4": applications.EfficientNetB4,
    "mobilenet": applications.MobileNet,
}

OPTIMIZERS = {
    "adam": optimizers.Adam,
    "sgd": optimizers.SGD,
    "rmsprop": optimizers.RMSprop,
    "lion": optimizers.Lion,
    "adamw": optimizers.AdamW # alternative to adam with l2 regularization
}

DENSE_ACTIVATION_FUNCTIONS = {
    "relu": activations.relu,
    "leaky_relu": activations.leaky_relu,
    "silu": activations.swish,
}

# Batch Normalization parameters
USE_BATCH_NORM = [True, False]
BN_MOMENTUM = [0.99, 0.9]
BN_EPSILON = [1e-3, 1e-4]

# Dropout parameters
DROPOUT_RATES = [0.0, 0.2, 0.4, 0.6]

# Weight Decay parameters (L2 regularization)
WEIGHT_DECAY = [0.0, 1e-5, 1e-4]

def generateModelName(file_extension: str, root_dir: str = ".", **kwargs) -> str:
    import os
    expected_keys = [
        "BASE_MODEL_NAME",
        "DENSE_ACTIVATION_FUNCTION",
        "USE_BATCH_NORM",
        "DROPOUT_RATE",
        "WEIGHT_DECAY",
        "LR_INIT",
        "LR_DECAY",
        "BASE_MODELS",
        "OPTIMIZER",
        "USE_VERSION",
    ]

    # Adjust expected keys based on conditional parameters
    if kwargs.get("USE_BATCH_NORM"):
        expected_keys += ["BN_MOMENTUM", "BN_EPSILON"]
    if kwargs.get("DROPOUT_RATE", 0.0) > 0.0:
        expected_keys.append("DROPOUT_RATE")
    if kwargs.get("USE_VERSION", False):
        expected_keys += ["DENSE_LAYER_SIZE", "NUMBER_OF_DENSE_LAYERS"]
    if kwargs.get("OPTIMIZER") == "adamw":
        expected_keys.append("ADAMW_DECAY")

    missing_keys = [key for key in expected_keys if key not in kwargs]
    unexpected_keys = [key for key in kwargs if key not in expected_keys]

    if missing_keys:
        logging.error(f"Missing keys in generateModelName: {missing_keys}")
    if unexpected_keys:
        logging.error(f"Unexpected keys in generateModelName: {unexpected_keys}")

    # Build name with only relevant parameters
    name_parts = [
        kwargs.get("BASE_MODEL_NAME", "UNKNOWN"),
        kwargs.get("DENSE_ACTIVATION_FUNCTION", "UNKNOWN"),
        kwargs.get("OPTIMIZER", "UNKNOWN"),
    ]

    if kwargs.get("USE_VERSION", 0) == 0:
        name_parts.extend([
            "ds" + str(kwargs.get("DENSE_LAYER_SIZE", "UNKNOWN")),
            "dl" + str(kwargs.get("NUMBER_OF_DENSE_LAYERS", "UNKNOWN")),
        ])

    name_parts.extend([
        "bn" if kwargs.get("USE_BATCH_NORM", False) else "nobn",
        "do" + str(kwargs.get("DROPOUT_RATE", 0.0)),
        "wd" + str(kwargs.get("WEIGHT_DECAY", 0.0)),
        "lr" + str(kwargs.get("LR_INIT", 0.0)),
        "decay" + str(kwargs.get("LR_DECAY", 0.0)),
        "adamw_decay" if kwargs.get("OPTIMIZER") == "adamw" else "",
    ])

    if kwargs.get("USE_BATCH_NORM", False):
        name_parts.extend([
            "bnmom" + str(kwargs.get("BN_MOMENTUM", "UNK")),
            "bneps" + str(kwargs.get("BN_EPSILON", "UNK")),
        ])

    if kwargs.get("OPTIMIZER") == "adamw":
        name_parts.append("adamw_decay" + str(kwargs.get("ADAMW_DECAY", "UNK")))

    name = "_".join(name_parts)
    if name.startswith("_"):
        name = name[1:]

    prefix_number = 0
    name_with_extension = f"{name}{file_extension}"
    while os.path.exists(os.path.join(root_dir, f"{prefix_number}_{name_with_extension}")):
        prefix_number += 1

    return f"{prefix_number}_{name_with_extension}"
