import os
import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm

REAL_PATH = "dataset/images"
GENERATED_PATH = "generated_images"


def load_images(folder, limit=20):

    images = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):

            path = os.path.join(folder, filename)

            image = tf.keras.utils.load_img(
                path,
                target_size=(128, 128)
            )

            image = tf.keras.utils.img_to_array(image)

            images.append(image)

            if len(images) >= limit:
                break

    return np.array(images, dtype=np.float32)


def calculate_fid(real_images, generated_images):

    real_images = real_images / 255.0
    generated_images = generated_images / 255.0

    real_features = real_images.reshape(
        real_images.shape[0], -1
    )

    generated_features = generated_images.reshape(
        generated_images.shape[0], -1
    )

    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(generated_features, axis=0)

    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(generated_features, rowvar=False)

    diff = mu1 - mu2

    covmean = sqrtm(
        sigma1 @ sigma2
    )

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = (
        diff @ diff
        + np.trace(
            sigma1 + sigma2 - 2 * covmean
        )
    )

    return float(fid)


print("=" * 50)
print("GAN FID EVALUATION")
print("=" * 50)

real_images = load_images(
    REAL_PATH,
    limit=20
)

generated_images = load_images(
    GENERATED_PATH,
    limit=20
)

print("Real Images      :", len(real_images))
print("Generated Images :", len(generated_images))

if len(real_images) < 2 or len(generated_images) < 2:

    print("Not enough images for FID calculation.")

else:

    fid_score = calculate_fid(
        real_images,
        generated_images
    )

    print("-" * 50)
    print("FID Score :", round(fid_score, 4))
    print("-" * 50)

print("=" * 50)
print("FID Evaluation Completed")
print("=" * 50)