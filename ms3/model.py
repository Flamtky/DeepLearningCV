import keras
from keras import layers
from keras import ops
import tensorflow as tf
import numpy as np

# Match constants with data_loader.py
AGE_GROUPS = [
    "INFANT",     # 0-4
    "CHILD",      # 5-12
    "TEENAGER",   # 13-19
    "YOUNG_ADULT",# 20-29
    "ADULT",      # 30-49
    "MIDDLE_AGED",# 50-59
    "SENIOR",     # 60+
]

# Constants
IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_AGE_CLASSES = len(AGE_GROUPS) 
NUM_GENDER_CLASSES = 2  # Only Male/Female
LATENT_DIM = 128

# Create the discriminator with multiple outputs
def build_discriminator():
    input_img = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    
    # Reduce initial filters and add dropout
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same")(input_img)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.GlobalAveragePooling2D()(x)

    # Add dense layers before outputs
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    
    validity = layers.Dense(1, name="validity")(x)
    
    return keras.Model(input_img, validity)

def build_generator():
    # Separate inputs for noise and conditions
    noise = layers.Input(shape=(LATENT_DIM,))
    age_input = layers.Input(shape=(NUM_AGE_CLASSES,))
    gender_input = layers.Input(shape=(NUM_GENDER_CLASSES,))
    
    # Concatenate noise and conditions
    x = layers.Concatenate()([noise, age_input, gender_input])
    
    # Increased initial dense size
    x = layers.Dense(14 * 14 * 512)(x)
    x = layers.Reshape((14, 14, 512))(x)
    
    x = layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Final conv with tanh
    x = layers.Conv2D(NUM_CHANNELS, (4, 4), padding="same", activation="tanh")(x)
    
    return keras.Model([noise, age_input, gender_input], x)

class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        # Unpack the data
        real_images, labels = data
        age_labels = labels["age"]
        gender_labels = labels["gender"]
        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector and conditions
                fake_images = self.generator(
                    [random_latent_vectors, age_labels, gender_labels], 
                    training=True
                )
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using conditions
            generated_images = self.generator(
                [random_latent_vectors, age_labels, gender_labels], 
                training=True
            )
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}