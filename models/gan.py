import tensorflow as tf
from models.generator import build_generator
from models.discriminator import build_discriminator

LATENT_DIM = 100

generator = build_generator()
discriminator = build_discriminator()

# Compile Discriminator
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Freeze discriminator while training GAN
discriminator.trainable = False

# GAN Model
gan = tf.keras.Sequential([
    generator,
    discriminator
])

gan.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss="binary_crossentropy"
)

if __name__ == "__main__":
    gan.summary()