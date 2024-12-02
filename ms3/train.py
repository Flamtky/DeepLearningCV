import keras
import os
from glob import glob
from model import WGAN, build_generator, build_discriminator, AGE_GROUPS, LATENT_DIM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_data
import pandas as pd
from evaluate import calculate_fid

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = (
    "false"  # True to prevent tensorflow from allocating all GPU memory
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Using GPU 1

class GenerateImagesCallback(keras.callbacks.Callback):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        # Fixed latent vectors and conditions for consistent visualization
        self.test_latent = tf.random.normal([len(AGE_GROUPS) * 2, latent_dim])
        
        # Create fixed conditions - 2 of each age group with alternating genders
        age_conditions = np.zeros((len(AGE_GROUPS) * 2, len(AGE_GROUPS)))
        gender_conditions = np.zeros((len(AGE_GROUPS) * 2, 2))
        
        for i in range(len(AGE_GROUPS)):  # age groups, 2 examples each
            age_conditions[i*2:(i+1)*2, i] = 1  # Set age group
            gender_conditions[i*2, 0] = 1       # Male
            gender_conditions[i*2+1, 1] = 1     # Female
            
        self.test_age = tf.constant(age_conditions, dtype=tf.float32)
        self.test_gender = tf.constant(gender_conditions, dtype=tf.float32)
        
    def on_epoch_end(self, epoch, logs=None):
        # Generate images
        generated_images = self.model.generator([self.test_latent, self.test_age, self.test_gender])
        
        # Plot results
        plt.figure(figsize=(12, 6))
        for i in range(len(AGE_GROUPS) * 2):
            plt.subplot(2, len(AGE_GROUPS), i+1)
            # Adjust image range from [-1,1] to [0,1]
            plt.imshow((generated_images[i] + 1) * 0.5)
            plt.axis('off')
            age_group = AGE_GROUPS[tf.argmax(self.test_age[i])]
            gender = "M" if self.test_gender[i][0] > 0 else "F"
            plt.title(f'{age_group}\n{gender}', fontsize=8)
        
        plt.suptitle(f'Epoch {epoch+1}')
        plt.tight_layout()
        # Save the figure
        plt.savefig(f'progress/epoch_{epoch+1:03d}.png')
        plt.close()

class EvaluateCallback(keras.callbacks.Callback):
    """Custom callback to evaluate FID score on validation data"""
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.results = {'epoch': [], 'fid': []}
        self.csv_path = 'logs/fid_scores.csv'
        
    def on_epoch_end(self, epoch, logs=None):
        real_images = []
        labels = []
        # 3 batches
        for _ in range(3):
            x, y = next(iter(self.validation_data))
            real_images.append(x)
            labels.append(y)
        print("Calculating FID score... for {len(real_images)} batches")
        real_images = tf.concat(real_images, axis=0)
        
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.model.latent_dim])
        
        age_labels = tf.concat([l['age'] for l in labels], axis=0)
        gender_labels = tf.concat([l['gender'] for l in labels], axis=0)
        combined_noise = tf.concat([noise, age_labels, gender_labels], axis=1)
        
        generated_images = self.model.generator(combined_noise)
        
        fid_score = calculate_fid(real_images, generated_images)
        
        self.results['epoch'].append(epoch + 1)
        self.results['fid'].append(fid_score)
        
        pd.DataFrame(self.results).to_csv(self.csv_path, index=False)
        
        logs = logs or {}
        logs['fid'] = fid_score

def find_latest_checkpoint():
    """Find the latest checkpoint in the checkpoints directory"""
    checkpoints = glob("checkpoints/cgan_*.weights.h5")
    if not checkpoints:
        return None, 0 # No checkpoints found
    
    epochs = []
    for ckpt in checkpoints:
        try:
            epoch = int(ckpt.split('_')[-1].split('.')[0])
            epochs.append(epoch)
        except ValueError:
            continue # Ignore checkpoints with non-integer epoch values (e.g., "best")
    latest_epoch = max(epochs)
    return f"checkpoints/cgan_{latest_epoch:02d}.weights.h5", latest_epoch

# Add WGAN loss functions
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

def train_cgan(epochs=100, batch_size=32, initial_epoch=0):
    os.makedirs('progress', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True) 
    
    # Load data
    train_dataset = load_data('', 'train', batch_size=batch_size)
    val_dataset = load_data('', 'val', batch_size=batch_size)

    # Initialize WGAN components
    discriminator = build_discriminator()
    generator = build_generator()
    
    # Initialize WGAN
    wgan = WGAN(
        discriminator=discriminator,
        generator=generator,
        latent_dim=LATENT_DIM,
        discriminator_extra_steps=5,  # Number of discriminator training steps
        gp_weight=10.0,  # Gradient penalty weight
    )

    # WGAN typically uses a lower learning rate
    LEARNING_RATE = 0.0001
    BETA_1 = 0.5
    BETA_2 = 0.9

    generator_optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, 
        beta_1=BETA_1, 
        beta_2=BETA_2
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1,
        beta_2=BETA_2
    )
    
    # Compile WGAN
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss,
    )

    latest_checkpoint, loaded_epoch = find_latest_checkpoint()
    
    # Only load checkpoint if it exists and we're not specifying an initial epoch
    if latest_checkpoint is not None and initial_epoch == 0:
        try:
            print(f"Attempting to load checkpoint: {latest_checkpoint}")
            wgan.load_weights(latest_checkpoint)
            initial_epoch = loaded_epoch
            print(f"Successfully resumed from epoch {initial_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            initial_epoch = 0
    else:
        print("No checkpoint found or initial epoch specified. Starting from scratch...")

    # Update callbacks to use wgan instead of cgan
    history = wgan.fit(
        train_dataset,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath="checkpoints/wgan_{epoch:02d}.weights.h5",
                save_weights_only=True,
                save_freq='epoch'
            ),
            keras.callbacks.ModelCheckpoint(
                filepath="checkpoints/wgan_best.weights.h5",
                save_weights_only=True,
                save_best_only=True,
                monitor='g_loss',
                mode='min'
            ),
            keras.callbacks.TensorBoard(
                log_dir='logs',
                update_freq='epoch'
            ),
            GenerateImagesCallback(),
            #EvaluateCallback(val_dataset)
        ]
    )
    return history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume-epoch', type=int, default=0)
    args = parser.parse_args()
    
    try:
        train_cgan(
            epochs=args.epochs,
            batch_size=args.batch_size,
            initial_epoch=args.resume_epoch
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress has been saved.")
        print("To resume, run with: --resume-epoch LAST_EPOCH")