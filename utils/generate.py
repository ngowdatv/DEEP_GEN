import os
import numpy as np
import tensorflow as tf

LATENT_DIM = 100

# Load trained generator
generator = tf.keras.models.load_model("saved_models/generator.keras")

# Create output folder
os.makedirs("generated_images", exist_ok=True)

# Generate random noise
noise = np.random.normal(0, 1, (10, LATENT_DIM))

# Generate images
generated = generator.predict(noise, verbose=0)

# Convert from [-1,1] to [0,1]
generated = (generated + 1) / 2.0

# Save images
for i, img in enumerate(generated):
    tf.keras.utils.save_img(
        f"generated_images/generated_{i+1}.png",
        img
    )

print("10 Images Generated Successfully!")