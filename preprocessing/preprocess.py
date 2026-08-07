import os
import tensorflow as tf

# Dataset path
DATASET_PATH = "dataset/images"

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32


def load_image_dataset():
    """
    Load image dataset from the dataset/images folder.
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        labels=None,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Normalize images (0-255 → 0-1)
    dataset = dataset.map(lambda x: x / 255.0)

    return dataset


if __name__ == "__main__":
    dataset = load_image_dataset()
    print("Dataset Loaded Successfully!")

    for batch in dataset.take(1):
        print("Batch Shape:", batch.shape)