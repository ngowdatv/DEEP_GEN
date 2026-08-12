import os
import random


def load_text(input_path):

    if not os.path.exists(input_path):
        print("ERROR: Text dataset not found.")
        print(input_path)
        return ""

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    return text


def generate_synthetic_text(
    input_path,
    output_path="generated_text/synthetic_text.txt",
    number_of_samples=20,
    words_per_sample=50
):

    print("=" * 60)
    print("TEXT DATA GENERATION")
    print("=" * 60)

    print("Text Dataset:")
    print(input_path)

    # --------------------------------------------------------
    # LOAD TEXT DATASET
    # --------------------------------------------------------

    print("-" * 60)
    print("LOADING TEXT DATASET")
    print("-" * 60)

    text = load_text(input_path)

    if not text:
        return False

    print("Text dataset loaded successfully.")

    # --------------------------------------------------------
    # PREPROCESS TEXT
    # --------------------------------------------------------

    words = text.split()

    print("Total words:", len(words))

    if len(words) < 2:

        print("ERROR: Text dataset is too small.")

        return False

    # --------------------------------------------------------
    # GENERATE SYNTHETIC TEXT
    # --------------------------------------------------------

    print("-" * 60)
    print("GENERATING SYNTHETIC TEXT")
    print("-" * 60)

    synthetic_samples = []

    for _ in range(number_of_samples):

        sample_words = []

        current_index = random.randint(
            0,
            len(words) - 1
        )

        for _ in range(words_per_sample):

            sample_words.append(
                words[current_index]
            )

            current_index = (
                current_index + random.randint(1, 3)
            ) % len(words)

        sample = " ".join(
            sample_words
        )

        synthetic_samples.append(
            sample
        )

    # --------------------------------------------------------
    # SAVE GENERATED TEXT
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for i, sample in enumerate(
            synthetic_samples,
            start=1
        ):

            file.write(
                f"Sample {i}:\n"
            )

            file.write(
                sample
            )

            file.write(
                "\n\n"
            )

    print("Synthetic text generated successfully.")

    print(
        "Generated Samples:",
        number_of_samples
    )

    print("Saved to:")
    print(output_path)

    print("=" * 60)
    print("TEXT GENERATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return True


if __name__ == "__main__":

    generate_synthetic_text(
        input_path="dataset/text/data.txt",
        output_path="generated_text/synthetic_text.txt",
        number_of_samples=20,
        words_per_sample=50
    )