import os
import numpy as np
import tensorflow as tf

LATENT_DIM = 100
NUMBER_OF_IMAGES = 20

MODEL_PATH = "saved_models/generator.keras"
OUTPUT_PATH = "generated_images"

os.makedirs(OUTPUT_PATH, exist_ok=True)

generator = tf.keras.models.load_model(MODEL_PATH)

noise = np.random.normal(
    0,
    1,
    (NUMBER_OF_IMAGES, LATENT_DIM)
)

generated_images = generator.predict(
    noise,
    verbose=1
)

generated_images = (
    generated_images + 1
) / 2.0

for i, image in enumerate(generated_images):

    filename = os.path.join(
        OUTPUT_PATH,
        f"synthetic_{i + 1}.png"
    )

    tf.keras.utils.save_img(
        filename,
        image
    )

print("=" * 40)
print("SYNTHETIC DATA GENERATION COMPLETE")
print("=" * 40)

print(
    f"Generated Images: {NUMBER_OF_IMAGES}"
)