import os

from utils.dataset_detector import detect_dataset_type


def run_pipeline(dataset_path):

    print("=" * 60)
    print("DEEP GENERATIVE FRAMEWORK")
    print("=" * 60)

    print("Uploaded Dataset:")
    print(dataset_path)

    # --------------------------------------------------------
    # STEP 1: DETECT DATASET
    # --------------------------------------------------------

    dataset_type = detect_dataset_type(dataset_path)

    print("-" * 60)
    print("Detected Dataset Type:", dataset_type)

    # --------------------------------------------------------
    # STEP 2: SELECT MODEL
    # --------------------------------------------------------

    if dataset_type.lower() == "image":

        print("-" * 60)
        print("Image dataset detected.")
        print("Selecting GAN model...")

        from training.train import train_gan

        train_gan(dataset_path)

        print("GAN processing completed.")

    elif dataset_type.lower() == "csv":

        print("-" * 60)
        print("CSV dataset detected.")
        print("Selecting CTGAN model...")

        from models.ctgan_model import train_ctgan

        train_ctgan(
            csv_path=dataset_path,
            output_path="generated_csv/synthetic_data.csv",
            epochs=100,
            num_rows=100
        )

        print("CTGAN processing completed.")

    elif dataset_type.lower() == "text":

        print("-" * 60)
        print("Text dataset detected.")
        print("Selecting Text Generation model...")

        from models.text_generator import generate_synthetic_text

        generate_synthetic_text(
            input_path=dataset_path,
            output_path="generated_text/synthetic_text.txt",
            number_of_samples=20,
            words_per_sample=50
        )

        print("Text generation completed.")

    else:

        print("-" * 60)
        print("ERROR: Unsupported dataset type.")
        return

    print("=" * 60)
    print("SYNTHETIC DATA GENERATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    path = input(
        "Enter uploaded dataset path: "
    )

    run_pipeline(path)
