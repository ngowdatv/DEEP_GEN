import os

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"
}

CSV_EXTENSIONS = {
    ".csv"
}

TEXT_EXTENSIONS = {
    ".txt", ".json"
}


def detect_dataset_type(dataset_path):

    image_count = 0
    csv_count = 0
    text_count = 0

    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            extension = os.path.splitext(file)[1].lower()

            if extension in IMAGE_EXTENSIONS:
                image_count += 1

            elif extension in CSV_EXTENSIONS:
                csv_count += 1

            elif extension in TEXT_EXTENSIONS:
                text_count += 1

    counts = {
        "image": image_count,
        "csv": csv_count,
        "text": text_count
    }

    dataset_type = max(counts, key=counts.get)

    if counts[dataset_type] == 0:
        return "unknown"

    return dataset_type


if __name__ == "__main__":

    path = "dataset"

    result = detect_dataset_type(path)

    print("=" * 40)
    print("DATASET DETECTION")
    print("=" * 40)
    print("Detected Type:", result.upper())