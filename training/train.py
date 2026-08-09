import os
import numpy as np
import tensorflow as tf

from models.generator import build_generator
from models.discriminator import build_discriminator

# =========================
# CONFIGURATION
# =========================

IMAGE_SIZE = 128
CHANNELS = 3
LATENT_DIM = 100

BATCH_SIZE = 16
EPOCHS = 5

DATASET_PATH = "dataset/images"

MODEL_PATH = "saved_models"
OUTPUT_PATH = "generated_images"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)


# =========================
# LOAD IMAGE DATASET
# =========================

print("=" * 50)
print("LOADING IMAGE DATASET")
print("=" * 50)

dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels=None,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Normalize images from [0,255] to [-1,1]
dataset = dataset.map(
    lambda x: (tf.cast(x, tf.float32) / 127.5) - 1
)

print("Dataset loaded successfully!")


# =========================
# BUILD MODELS
# =========================

generator = build_generator(LATENT_DIM)
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

g_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)

d_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)


# =========================
# TRAINING STEP
# =========================

@tf.function
def train_step(real_images):

    batch_size = tf.shape(real_images)[0]

    noise = tf.random.normal(
        [batch_size, LATENT_DIM]
    )

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

        fake_images = generator(
            noise,
            training=True
        )

        real_output = discriminator(
            real_images,
            training=True
        )

        fake_output = discriminator(
            fake_images,
            training=True
        )

        g_loss = cross_entropy(
            tf.ones_like(fake_output),
            fake_output
        )

        d_loss_real = cross_entropy(
            tf.ones_like(real_output),
            real_output
        )

        d_loss_fake = cross_entropy(
            tf.zeros_like(fake_output),
            fake_output
        )

        d_loss = (
            d_loss_real + d_loss_fake
        ) / 2

    g_gradients = g_tape.gradient(
        g_loss,
        generator.trainable_variables
    )

    d_gradients = d_tape.gradient(
        d_loss,
        discriminator.trainable_variables
    )

    g_optimizer.apply_gradients(
        zip(
            g_gradients,
            generator.trainable_variables
        )
    )

    d_optimizer.apply_gradients(
        zip(
            d_gradients,
            discriminator.trainable_variables
        )
    )

    return g_loss, d_loss


# =========================
# TRAIN GAN
# =========================

print("=" * 50)
print("STARTING GAN TRAINING")
print("=" * 50)

generator_losses = []
discriminator_losses = []

for epoch in range(EPOCHS):

    g_total = 0
    d_total = 0
    batches = 0

    for images in dataset:

        g_loss, d_loss = train_step(images)

        g_total += float(g_loss)
        d_total += float(d_loss)

        batches += 1

    g_average = g_total / batches
    d_average = d_total / batches

    generator_losses.append(g_average)
    discriminator_losses.append(d_average)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Generator Loss: {g_average:.4f} | "
        f"Discriminator Loss: {d_average:.4f}"
    )


# =========================
# SAVE MODELS
# =========================

generator.save(
    os.path.join(
        MODEL_PATH,
        "generator.keras"
    )
)

discriminator.save(
    os.path.join(
        MODEL_PATH,
        "discriminator.keras"
    )
)

print("=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)

print("Generator saved successfully!")
print("Discriminator saved successfully!")