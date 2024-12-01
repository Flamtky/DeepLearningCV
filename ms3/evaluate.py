import tensorflow as tf
from scipy.stats import entropy
import numpy as np

def calculate_fid(real_images, generated_images):
    """Calculate Fréchet Inception Distance"""
    inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')
    
    real_features = inception_model.predict(real_images)
    gen_features = inception_model.predict(generated_images)
    
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = tf.linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

"""
def evaluate_model(test_dataset):
    cgan = ConditionalGAN(discriminator, generator, LATENT_DIM)
    cgan.load_weights("checkpoints/cgan_best.keras")
    
    real_images = []
    generated_images = []
    
    for batch in test_dataset:
        real_batch = batch[0]
        real_images.append(real_batch)
        
        # Generate matching conditions
        z = tf.random.normal((len(real_batch), LATENT_DIM))
        generated_batch = cgan.generator.predict(z)
        generated_images.append(generated_batch)
    
    real_images = np.concatenate(real_images)
    generated_images = np.concatenate(generated_images)
    
    fid_score = calculate_fid(real_images, generated_images)
    print(f"FID Score: {fid_score}")
"""
