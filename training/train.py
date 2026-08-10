import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.generator import build_generator
from models.discriminator import build_discriminator


# ============================================================
# CONFIGURATION
# ============================================================

LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 5
IMAGE_SIZE = 128

DATASET_PATH = "dataset/images"
MODEL_PATH = "saved_models"
EVALUATION_PATH = "evaluation"
OUTPUT_PATH = "generated_images"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(EVALUATION_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ============================================================
# LOAD IMAGE DATASET
# ============================================================

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


def normalize(image):
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1.0


dataset = dataset.map(normalize)

print("Dataset loaded successfully!")


# ============================================================
# BUILD MODELS
# ============================================================

generator = build_generator()
discriminator = build_discriminator()

print("Generator loaded successfully")
print("Discriminator loaded successfully")


# ============================================================
# OPTIMIZERS
# ============================================================

cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=False
)

generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def generator_loss(fake_output):

    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )


def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    return real_loss + fake_loss


# ============================================================
# TRAINING STEP
# ============================================================

@tf.function
def train_step(images):

    noise = tf.random.normal(
        [tf.shape(images)[0], LATENT_DIM]
    )

    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as disc_tape:

        generated_images = generator(
            noise,
            training=True
        )

        real_output = discriminator(
            images,
            training=True
        )

        fake_output = discriminator(
            generated_images,
            training=True
        )

        gen_loss = generator_loss(fake_output)

        disc_loss = discriminator_loss(
            real_output,
            fake_output
        )

    gradients_of_generator = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    gradients_of_discriminator = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(
            gradients_of_generator,
            generator.trainable_variables
        )
    )

    discriminator_optimizer.apply_gradients(
        zip(
            gradients_of_discriminator,
            discriminator.trainable_variables
        )
    )

    return gen_loss, disc_loss


# ============================================================
# TRAIN GAN
# ============================================================

print("=" * 50)
print("STARTING GAN TRAINING")
print("=" * 50)

generator_losses = []
discriminator_losses = []


for epoch in range(EPOCHS):

    gen_epoch_loss = []
    disc_epoch_loss = []

    for images in dataset:

        gen_loss, disc_loss = train_step(images)

        gen_epoch_loss.append(
            float(gen_loss)
        )

        disc_epoch_loss.append(
            float(disc_loss)
        )

    # Calculate average losses for the epoch

    avg_gen_loss = np.mean(
        gen_epoch_loss
    )

    avg_disc_loss = np.mean(
        disc_epoch_loss
    )

    generator_losses.append(
        avg_gen_loss
    )

    discriminator_losses.append(
        avg_disc_loss
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Generator Loss: {avg_gen_loss:.4f} "
        f"- Discriminator Loss: {avg_disc_loss:.4f}"
    )

    # ========================================================
    # SAVE CHECKPOINT AFTER EVERY EPOCH
    # ========================================================

    generator.save(
        os.path.join(
            MODEL_PATH,
            f"generator_epoch_{epoch + 1}.keras"
        )
    )

    discriminator.save(
        os.path.join(
            MODEL_PATH,
            f"discriminator_epoch_{epoch + 1}.keras"
        )
    )

    print(
        f"Checkpoint saved for Epoch {epoch + 1}"
    )


# ============================================================
# TRAINING COMPLETED
# ============================================================

print("=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)


# ============================================================
# SAVE FINAL MODELS
# ============================================================

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

print("Generator saved successfully")
print("Discriminator saved successfully")


# ============================================================
# SAVE LOSS DATA
# ============================================================

np.save(
    os.path.join(
        EVALUATION_PATH,
        "generator_losses.npy"
    ),
    np.array(generator_losses)
)

np.save(
    os.path.join(
        EVALUATION_PATH,
        "discriminator_losses.npy"
    ),
    np.array(discriminator_losses)
)

print("Loss data saved successfully")


# ============================================================
# CREATE LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    generator_losses,
    label="Generator Loss"
)

plt.plot(
    discriminator_losses,
    label="Discriminator Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "GAN Training Loss"
)

plt.legend()

plt.grid(True)

plt.savefig(
    os.path.join(
        EVALUATION_PATH,
        "loss_graph.png"
    )
)

plt.close()

print("Loss graph saved successfully")
print(
    "Location: evaluation/loss_graph.png"
)


# ============================================================
# GENERATE SYNTHETIC IMAGES
# ============================================================

print("=" * 50)
print("GENERATING SYNTHETIC IMAGES")
print("=" * 50)

NUMBER_OF_IMAGES = 20

noise = tf.random.normal(
    [NUMBER_OF_IMAGES, LATENT_DIM]
)

generated_images = generator(
    noise,
    training=False
)


# Convert from [-1, 1] to [0, 255]

generated_images = (
    (generated_images + 1.0) * 127.5
)

generated_images = tf.clip_by_value(
    generated_images,
    0,
    255
)

generated_images = tf.cast(
    generated_images,
    tf.uint8
)


# ============================================================
# SAVE GENERATED IMAGES
# ============================================================

for i in range(NUMBER_OF_IMAGES):

    image = generated_images[i].numpy()

    tf.keras.utils.save_img(
        os.path.join(
            OUTPUT_PATH,
            f"synthetic_{i + 1}.png"
        ),
        image
    )


print("Generated Images:", NUMBER_OF_IMAGES)
print("Synthetic image generation completed")

print("=" * 50)
print("GAN PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 50)