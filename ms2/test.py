import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings

import keras
from data_loader import load_data
import model as mo
import parameters as pa

MODEL_NAME = "0_efficientnetb0_relu_ds512_dl2_nobn_do0.0_wd0.0_lr1e-05_decay0.9.keras"
MODEL_PATH = os.path.join("./models", MODEL_NAME)


def main():
    model: keras.Model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "face_loss": mo.face_loss,
            "conditional_age_loss": mo.conditional_age_loss,
            "conditional_gender_loss": mo.conditional_gender_loss,
            "age_acc": mo.age_acc,
            "gen_acc": mo.gen_acc,
        },
    )

    test_dataset = load_data(".", "test", batch_size=pa.BATCH_SIZE)

    results = model.evaluate(test_dataset, batch_size=pa.BATCH_SIZE)
    print("Test results:")
    for metric, value in zip(model.metrics_names, results):
        print(f"-{metric}: {value}")


if __name__ == "__main__":
    main()
