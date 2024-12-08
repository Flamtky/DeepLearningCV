import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = (
    "false"  # True to prevent tensorflow from allocating all GPU memory
)
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # GPU to use

import keras
from utils.data_loader import load_data_new
from model import (
    create_model,
    conditional_age_loss,
    conditional_gender_loss,
    face_loss,
    age_acc,
    gen_acc,
)
import parameters as pa


DATA_DIR = "."


def main(testname="", **kwargs):
    train_dataset = load_data_new(DATA_DIR, "train", batch_size=pa.BATCH_SIZE)
    val_dataset = load_data_new(DATA_DIR, "val", batch_size=pa.BATCH_SIZE)

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
        "USE_VERSION": 2, # 0 old version for dense layer testing; 1 is own custom composition; 2 is final model
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

def final_model():
    # Final Model
    # v2:
    # face: dense 128, leaky relu, ohne Dropout, Adam
    # age: 3x dense 512, leaky relu, dropout 0.2, Adam
    # gender: dense 512, leaky relu, dropout 0.2, Adam
    main(
        "ms3_final",
        USE_VERSION=2,
    )

if __name__ == "__main__":
    final_model()
