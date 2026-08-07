import tensorflow as tf
from tensorflow.keras import layers

IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3

def build_discriminator():

    model = tf.keras.Sequential(name="Discriminator")

    model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)))

    model.add(layers.Conv2D(64, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(layers.Dense(1, activation="sigmoid"))

    return model


if __name__ == "__main__":
    discriminator = build_discriminator()
    discriminator.summary()