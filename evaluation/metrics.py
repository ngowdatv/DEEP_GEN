import os

def count_generated_images(folder="generated_images"):
    """
    Count the number of generated images.
    """
    if not os.path.exists(folder):
        return 0

    images = [
        file for file in os.listdir(folder)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    return len(images)


if __name__ == "__main__":
    total = count_generated_images()
    print("=" * 40)
    print("GAN METRICS")
    print("=" * 40)
    print(f"Generated Images : {total}")