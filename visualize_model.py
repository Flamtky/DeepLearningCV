import visualkeras
from keras.models import load_model
import model as mo
import os

# Load the model
MODEL_PATH = (
    "./models/0_efficientnetb0_relu_nobn_do0.0_wd0.0_lr1e-05_decay0.9.keras"
)

model = load_model(
    MODEL_PATH,
    custom_objects={
        "face_loss": mo.face_loss,
        "conditional_age_loss": mo.conditional_age_loss,
        "conditional_gender_loss": mo.conditional_gender_loss,
        "age_acc": mo.age_acc,
        "gen_acc": mo.gen_acc,
    },
)

# Visualize the model
visualkeras.layered_view(model).show()
# Save the model visualization
visualkeras.layered_view(model, to_file="model.png").show()
