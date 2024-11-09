import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = (
    "false"  # True to prevent tensorflow from allocating all GPU memory
)

import keras
from data_loader import load_data
from model import (
    create_model,
    conditional_age_loss,
    conditional_gender_loss,
    face_loss,
    age_acc,
    gen_acc,
)
import parameters as pa


SHUTDOWN_AFTER_TRAINING = False
DATA_DIR = "."


def main(testname="", **kwargs):
    train_dataset = load_data(DATA_DIR, "train", batch_size=pa.BATCH_SIZE)
    val_dataset = load_data(DATA_DIR, "val", batch_size=pa.BATCH_SIZE)

    model_parameters = {
        "BASE_MODEL_NAME": "efficientnetb0",
        "DENSE_LAYER_SIZE": 512,
        "NUMBER_OF_DENSE_LAYERS": 2,
        "DENSE_ACTIVATION_FUNCTION": "relu",
        "USE_BATCH_NORM": False,
        "DROPOUT_RATE": 0.0,
        "WEIGHT_DECAY": 0.0,
        "LR_INIT": 1e-5,
        "LR_DECAY": 0.9,
        "BASE_MODELS": pa.BASE_MODELS,
        "OPTIMIZER": "adam",
        "ADAMW_DECAY": 1e-5,
        "USE_VERSION": 1, # 0 old version for dense layer testing; 1 is own custom composition; 2 is final model
    }

    # Update parameters with any provided kwargs
    model_parameters.update((k, v) for k, v in kwargs.items() if v is not None)


    # Debug Prints
    print(pa.generateModelName("", ".", **model_parameters))

    model = create_model(model_parameters, face_only=pa.FACE_ONLY)
    model.summary()

    # Optionally load weights from a checkpoint
    checkpoint_path = None
    if checkpoint_path is not None:
        model.load_weights(checkpoint_path)

    if pa.FACE_ONLY:
        metrics = {
            "face": ["accuracy"],
        }
        losses = {
            "face": face_loss,
        }
        pa.LOSS_WEIGHTS = {
            "face": 1.0,
        }
    else:
        metrics = {
            "face": ["accuracy"],
            "age": [age_acc],
            "gender": [gen_acc],
        }
        losses = {
            "face": face_loss,
            "age": conditional_age_loss,
            "gender": conditional_gender_loss,
        }

    if model_parameters["OPTIMIZER"] == "adamw":
        optimizer = pa.OPTIMIZERS[model_parameters["OPTIMIZER"]](
            learning_rate=model_parameters["LR_INIT"],
            weight_decay=model_parameters["ADAMW_DECAY"],
        )
    else:
        optimizer = pa.OPTIMIZERS[model_parameters["OPTIMIZER"]](
            model_parameters["LR_INIT"]
        )
    model.compile(
        optimizer=optimizer,
        loss_weights=pa.LOSS_WEIGHTS,
        metrics=metrics,
        loss=losses,
        run_eagerly=False,
    )

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/"
        + pa.generateModelName(".keras", "./checkpoints", **model_parameters),
        save_weights_only=False,
        save_freq="epoch",
        verbose=1,
        save_best_only=True,
    )

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=model_parameters["LR_DECAY"],
        patience=1,
        verbose=1,
        mode="auto",
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, verbose=1, restore_best_weights=True
    )

    tensorBoard_callback = keras.callbacks.TensorBoard(
        f"logs/{testname}/"
        + pa.generateModelName("", f"logs/{testname}", **model_parameters)
    )

    # Train the model
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=pa.EPOCHS,
        callbacks=[
            checkpoint_callback,
            tensorBoard_callback,
            lr_scheduler,
            # early_stopping
        ],
    )

    # Save the final model
    model.save(
        "models/" + pa.generateModelName(".keras", "./models", **model_parameters)
    )

    if SHUTDOWN_AFTER_TRAINING:
        os.system("/mnt/c/Windows/System32/cmd.exe /c shutdown /s /t 0")

"""Tests for testing Hyperparameters"""

def test1():
    # Find Dense Layer Size and Number of Dense Layers
    things_to_learn = {
        "DENSE_LAYER_SIZE": [128, 256, 512],
        "NUMBER_OF_DENSE_LAYERS": [1, 2, 3],
    }  # 2 -> 512 each
    for dense_layer_size in things_to_learn["DENSE_LAYER_SIZE"]:
        for number_of_dense_layers in things_to_learn["NUMBER_OF_DENSE_LAYERS"]:
            main(
                DENSE_LAYER_SIZE=dense_layer_size,
                NUMBER_OF_DENSE_LAYERS=number_of_dense_layers,
            )


def test2():
    # Test new composition of small and large dense layers
    main(USE_VERSION=1)


def test3():
    # Learnrate init and decay
    import math

    things_to_learn = {
        "LR_INIT": [1e-4, 1e-5],
        "LR_DECAY": [0.9, 0.95, 0.99],
    }
    for lr_init in things_to_learn["LR_INIT"]:
        for lr_decay in things_to_learn["LR_DECAY"]:
            if math.isclose(lr_decay, 0.9) and math.isclose(
                lr_init, 1e-5
            ):  # isclose to avoid floating point errors
                continue  # Already trained
            print(f"Training with lr_init={lr_init} and lr_decay={lr_decay}")
            main(
                "test3",
                LR_INIT=lr_init,
                LR_DECAY=lr_decay,
            )


def test4():
    # Base model
    BASE_MODELS_TO_TEST = [
        "efficientnetb4",
        "mobilenet",
    ]  # skip b0 as it is already trained

    # Testing custom composition of dense layers (test2) with different base models
    for base_model in BASE_MODELS_TO_TEST:
        main("test4", BASE_MODEL_NAME=base_model)

    things_to_learn = {
        "DENSE_LAYER_SIZE": [128, 256, 512],
        "NUMBER_OF_DENSE_LAYERS": [1, 2],
    }
    for base_model in BASE_MODELS_TO_TEST:
        for dense_layer_size in things_to_learn["DENSE_LAYER_SIZE"]:
            for number_of_dense_layers in things_to_learn["NUMBER_OF_DENSE_LAYERS"]:
                main(
                    "test4",
                    BASE_MODEL_NAME=base_model,
                    DENSE_LAYER_SIZE=dense_layer_size,
                    NUMBER_OF_DENSE_LAYERS=number_of_dense_layers,
                    USE_VERSION=0,  # Use defined dense layers instead of custom composition
                )


# ETA: 2*4*2h = 16h
def test5():
    # Optimizer
    # Adam, RMSprop, SGD, Lion
    things_to_learn = {
        "BASE_MODEL_NAME": ["efficientnetb0", "mobilenet"],
        "OPTIMIZER": ["adam", "rmsprop", "sgd", "lion"],
    }
    for base_model in things_to_learn["BASE_MODEL_NAME"]:
        for optimizer in things_to_learn["OPTIMIZER"]:
            main(
                "test5",
                BASE_MODEL_NAME=base_model,
                OPTIMIZER=optimizer,
            )


def test6():
    # Activation function
    # relu, leaky_relu, silu
    things_to_learn = {
        "BASE_MODEL_NAME": ["efficientnetb0", "mobilenet"],
        "DENSE_ACTIVATION_FUNCTION": ["leaky_relu", "silu", "relu"],
    }
    for base_model in things_to_learn["BASE_MODEL_NAME"]:
        for activation_function in things_to_learn["DENSE_ACTIVATION_FUNCTION"]:
            main(
                "test6",
                BASE_MODEL_NAME=base_model,
                DENSE_ACTIVATION_FUNCTION=activation_function,
            )


def test7():
    # Dropout
    things_to_learn = {
        "BASE_MODEL_NAME": ["efficientnetb0", "mobilenet"],
        "DROPOUT_RATE": [0.2, 0.4],
    }
    for base_model in things_to_learn["BASE_MODEL_NAME"]:
        for dropout_rate in things_to_learn["DROPOUT_RATE"]:
            main(
                "test7",
                BASE_MODEL_NAME=base_model,
                DROPOUT_RATE=dropout_rate,
                DENSE_ACTIVATION_FUNCTION="leaky_relu",
            )

def test8():
    # Batch Normalization (with and without Dropout)
    BASE_MODEL = "efficientnetb0"
    things_to_learn = {
        "USE_BATCH_NORM": [True],
        "DROPOUT_RATE": [0.0, 0.2],
    }
    for use_batch_norm in things_to_learn["USE_BATCH_NORM"]:
        for dropout_rate in things_to_learn["DROPOUT_RATE"]:
            main(
                "test8",
                BASE_MODEL_NAME=BASE_MODEL,
                USE_BATCH_NORM=use_batch_norm,
                DENSE_ACTIVATION_FUNCTION="leaky_relu",
                DROPOUT_RATE=dropout_rate,
            )


def test9():
    # Weight Decay
    BASE_MODEL = "efficientnetb0"
    things_to_learn = {
        "WEIGHT_DECAY": [1e-4, 1e-5],
    }
    for weight_decay in things_to_learn["WEIGHT_DECAY"]:
        main(
            "test9",
            BASE_MODEL_NAME=BASE_MODEL,
            WEIGHT_DECAY=weight_decay,
            DENSE_ACTIVATION_FUNCTION="leaky_relu",
        )
    # Alternative adamw
    for weight_decay in things_to_learn["WEIGHT_DECAY"]:
        main(
            "test9",
            OPTIMIZER="adamw",
            ADAMW_DECAY=weight_decay,
            DENSE_ACTIVATION_FUNCTION="leaky_relu",
        )

def test10():
    # Final Model
    # v2:
    # face: dense 128, leaky relu, ohne Dropout, AdamW
    # age: 3x dense 512, leaky relu, dropout 0.2, AdamW
    # gender: dense 512, leaky relu, dropout 0.2, AdamW
    BASE_MODEL = "efficientnetb0"
    main(
        "test10",
        USE_VERSION=2,
        OPTIMIZER="adamw", # Rest is set in v2 method
    )

if __name__ == "__main__":
    test10()
