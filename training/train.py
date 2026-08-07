import tensorflow as tf
import numpy as np

from models.generator import build_generator
from models.discriminator import build_discriminator

LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 5

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

discriminator.trainable = False

gan = tf.keras.Sequential([
    generator,
    discriminator
])

gan.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
    loss="binary_crossentropy"
)

print("=" * 50)
print("Starting GAN Training...")
print("=" * 50)

for epoch in range(EPOCHS):

    # Random noise
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

    # Generate fake images
    fake_images = generator.predict(noise, verbose=0)

    # Dummy real images (replace with your dataset later)
    real_images = np.random.rand(BATCH_SIZE, 128, 128, 3)

    real_labels = np.ones((BATCH_SIZE, 1))
    fake_labels = np.zeros((BATCH_SIZE, 1))

    discriminator.trainable = True

    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

    discriminator.trainable = False

    g_loss = gan.train_on_batch(noise, real_labels)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Discriminator Real Loss : {d_loss_real}")
    print(f"Discriminator Fake Loss : {d_loss_fake}")
    print(f"Generator Loss          : {g_loss}")
    print("-" * 50)

print("Training Completed Successfully!")

generator.save("saved_models/generator.keras")
discriminator.save("saved_models/discriminator.keras")

print("Models saved successfully!")