import os
import numpy as np
import tensorflow as tf


def create_directory(path):
    """
    Create directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def generate_noise(batch_size, latent_dim):
    """
    Generate random latent vectors.
    """
    return np.random.normal(0, 1, (batch_size, latent_dim))


def save_generated_images(generator, latent_dim, num_images=10,
                          output_dir="generated_images"):

    create_directory(output_dir)

    noise = generate_noise(num_images, latent_dim)

    generated_images = generator.predict(noise, verbose=0)

    generated_images = (generated_images + 1) / 2.0

    for i, image in enumerate(generated_images):
        tf.keras.utils.save_img(
            os.path.join(output_dir, f"generated_{i+1}.png"),
            image
        )

    print(f"{num_images} images saved in '{output_dir}'")


def save_model(model, path):
    """
    Save a trained model.
    """
    model.save(path)
    print(f"Model saved to {path}")