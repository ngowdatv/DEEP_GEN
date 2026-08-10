import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


REAL_PATH = "dataset/images"
GENERATED_PATH = "generated_images"


def load_images(folder):

    images = []

    if not os.path.exists(folder):
        return images

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):

            path = os.path.join(folder, filename)

            try:
                image = Image.open(path).convert("L")
                image = image.resize((128, 128))

                images.append(np.array(image))

            except Exception as e:
                print("Could not load:", filename)
                print("Error:", e)

    return images


print("=" * 50)
print("GAN IMAGE EVALUATION")
print("=" * 50)


real_images = load_images(REAL_PATH)
generated_images = load_images(GENERATED_PATH)


print("Real Images      :", len(real_images))
print("Generated Images :", len(generated_images))


if len(real_images) == 0:

    print("ERROR: No real images found.")
    print("Check:", REAL_PATH)


elif len(generated_images) == 0:

    print("ERROR: No generated images found.")
    print("Check:", GENERATED_PATH)


else:

    scores = []

    count = min(
        len(real_images),
        len(generated_images)
    )

    for i in range(count):

        score = ssim(
            real_images[i],
            generated_images[i],
            data_range=255
        )

        scores.append(score)

    average_ssim = np.mean(scores)

    print("-" * 50)
    print("Images Compared :", count)
    print("Average SSIM    :", round(average_ssim, 4))
    print("-" * 50)


    if average_ssim >= 0.70:

        print("Image Similarity: GOOD")

    elif average_ssim >= 0.40:

        print("Image Similarity: MODERATE")

    else:

        print("Image Similarity: LOW")


print("=" * 50)
print("Evaluation Completed")
print("=" * 50)