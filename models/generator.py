import tensorflow as tf


LATENT_DIM = 100


def build_generator():

    model = tf.keras.Sequential(
        name="Generator"
    )

    model.add(
        tf.keras.layers.Input(
            shape=(LATENT_DIM,)
        )
    )

    model.add(
        tf.keras.layers.Dense(
            8 * 8 * 256
        )
    )

    model.add(
        tf.keras.layers.BatchNormalization()
    )

    model.add(
        tf.keras.layers.LeakyReLU()
    )

    model.add(
        tf.keras.layers.Reshape(
            (8, 8, 256)
        )
    )

    model.add(
        tf.keras.layers.Conv2DTranspose(
            128,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(
        tf.keras.layers.BatchNormalization()
    )

    model.add(
        tf.keras.layers.LeakyReLU()
    )

    model.add(
        tf.keras.layers.Conv2DTranspose(
            64,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(
        tf.keras.layers.BatchNormalization()
    )

    model.add(
        tf.keras.layers.LeakyReLU()
    )

    model.add(
        tf.keras.layers.Conv2DTranspose(
            32,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(
        tf.keras.layers.BatchNormalization()
    )

    model.add(
        tf.keras.layers.LeakyReLU()
    )

    model.add(
        tf.keras.layers.Conv2DTranspose(
            3,
            kernel_size=4,
            strides=2,
            padding="same",
            activation="tanh"
        )
    )

    return model


if __name__ == "__main__":

    generator = build_generator()

    generator.summary()