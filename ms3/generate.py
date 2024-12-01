import numpy as np
import tensorflow as tf
from model import ConditionalGAN, generator, discriminator, LATENT_DIM, AGE_GROUPS
import matplotlib.pyplot as plt

random = np.random.Generator(1337)

def generate_images(n_images=10, age_group=None, gender=None):
    # Load trained model
    cgan = ConditionalGAN(discriminator, generator, LATENT_DIM)
    cgan.load_weights("checkpoints/cgan_best.keras")
    
    # Generate random latent vectors
    latent_vectors = tf.random.normal((n_images, LATENT_DIM))
    
    # Create conditions
    age_conditions = np.zeros((n_images, len(AGE_GROUPS)))
    if age_group is not None:
        age_conditions[:, age_group] = 1
    else:
        age_group = random.randint(0, len(AGE_GROUPS), n_images)
        age_conditions[np.arange(n_images), age_group] = 1
        
    gender_conditions = np.zeros((n_images, 2))
    if gender is not None:
        gender_conditions[:, gender] = 1
    else:
        gender = randomrandint(0, 2, n_images)
        gender_conditions[np.arange(n_images), gender] = 1
    
    # Combine inputs
    generator_inputs = tf.concat(
        [latent_vectors, age_conditions, gender_conditions], 
        axis=1
    )
    
    # Generate images
    generated_images = cgan.generator(generator_inputs)
    
    # Plot results
    _, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, img in enumerate(generated_images):
        ax = axes[i//5, i%5]
        ax.imshow((img * 0.5 + 0.5))
        ax.axis('off')
        ax.set_title(f'Age: {AGE_GROUPS[age_group[i]]}\nGender: {"M" if gender[i]==0 else "F"}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_images()
