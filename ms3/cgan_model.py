import keras
from keras import layers
from keras import ops
import tensorflow as tf
import numpy as np

AGE_GROUPS = [
    "INFANT",     # 0-4
    "CHILD",      # 5-12
    "TEENAGER",   # 13-19
    "YOUNG_ADULT",# 20-29
    "ADULT",      # 30-49
    "MIDDLE_AGED",# 50-59
    "SENIOR",     # 60+
]

IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_AGE_CLASSES = len(AGE_GROUPS)
NUM_GENDER_CLASSES = 2
LATENT_DIM = 128

# Discriminator with multiple outputs
def build_discriminator():
    input_img = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    
    # Add batch normalization and increase filters
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.GlobalAveragePooling2D()(x) 
    x = layers.Dropout(0.3)(x)
    
    # Add dense layers before outputs
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    
    validity = layers.Dense(1, activation="sigmoid", name="validity")(x)
    age_output = layers.Dense(NUM_AGE_CLASSES, activation="softmax", name="age")(x)
    gender_output = layers.Dense(NUM_GENDER_CLASSES, activation="softmax", name="gender")(x)
    
    return keras.Model(input_img, [validity, age_output, gender_output])

def build_generator():
    noise_shape = (LATENT_DIM + NUM_AGE_CLASSES + NUM_GENDER_CLASSES,)
    noise = layers.Input(shape=noise_shape)
    
    # Increased initial dense size
    x = layers.Dense(14 * 14 * 512)(noise)
    x = layers.Reshape((14, 14, 512))(x)
    
    x = layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Final conv with tanh
    x = layers.Conv2D(NUM_CHANNELS, (5, 5), padding="same", activation="tanh")(x)
    
    return keras.Model(noise, x)

class ConditionalGAN(keras.Model):
    def __init__(self):
        super().__init__()
        self.discriminator = build_discriminator()
        self.generator = build_generator()

        # Summary
        self.discriminator.summary()
        self.generator.summary()

        self.latent_dim = LATENT_DIM
        self.seed_generator = keras.random.SeedGenerator(1337)
        
        # Loss trackers
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.age_loss_tracker = keras.metrics.Mean(name="age_loss")
        self.gender_loss_tracker = keras.metrics.Mean(name="gender_loss")

    @property
    def metrics(self):
        return [
            self.gen_loss_tracker,
            self.disc_loss_tracker,
            self.age_loss_tracker,
            self.gender_loss_tracker,
        ]

    def compile(self, d_optimizer, g_optimizer, discriminator_extra_steps=3):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.discriminator_extra_steps = discriminator_extra_steps
        
        self.binary_loss = keras.losses.BinaryCrossentropy()
        self.categorical_loss = keras.losses.CategoricalCrossentropy()
        
        # Add gradient penalty
        self.gp_weight = 10.0

    def train_step(self, data):
        # Unpack the data
        print(data)
        real_images, labels = data
        batch_size = tf.shape(real_images)[0]
        
        # Extract labels
        age_labels = labels["age"]
        gender_labels = labels["gender"]
        
        # Train discriminator
        noise = tf.random.normal([batch_size, self.latent_dim])
        combined_noise = tf.concat([noise, age_labels, gender_labels], axis=1)
        generated_images = self.generator(combined_noise, training=True)
        
        with tf.GradientTape() as tape:
            # Real images
            real_validity, real_age, real_gender = self.discriminator(real_images, training=True)
            # Fake images
            fake_validity, fake_age, fake_gender = self.discriminator(generated_images, training=True)
            
            # Calculate losses
            d_loss_real = self.binary_loss(tf.ones_like(real_validity), real_validity)
            d_loss_fake = self.binary_loss(tf.zeros_like(fake_validity), fake_validity)
            d_loss_age = self.categorical_loss(age_labels, real_age)
            d_loss_gender = self.categorical_loss(gender_labels, real_gender)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake + d_loss_age + d_loss_gender
            
        # Train discriminator
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Train generator
        noise = tf.random.normal([batch_size, self.latent_dim])
        combined_noise = tf.concat([noise, age_labels, gender_labels], axis=1)
        
        with tf.GradientTape() as tape:
            generated_images = self.generator(combined_noise, training=True)
            fake_validity, fake_age, fake_gender = self.discriminator(generated_images, training=True)
            
            # Calculate losses
            g_loss_fake = self.binary_loss(tf.ones_like(fake_validity), fake_validity)
            g_loss_age = self.categorical_loss(age_labels, fake_age)
            g_loss_gender = self.categorical_loss(gender_labels, fake_gender)
            
            # Total generator loss
            g_loss = g_loss_fake + g_loss_age + g_loss_gender
            
        # Train generator
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Update metrics
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        self.age_loss_tracker.update_state(d_loss_age)
        self.gender_loss_tracker.update_state(d_loss_gender)
        
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "age_loss": self.age_loss_tracker.result(),
            "gender_loss": self.gender_loss_tracker.result(),
        }
    
    def call(self, inputs):
        return self.generator(inputs)