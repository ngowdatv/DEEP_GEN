from models.ctgan_model import train_ctgan


CSV_PATH = "dataset/csv/data.csv"

OUTPUT_PATH = "generated_csv/synthetic_data.csv"


if __name__ == "__main__":

    train_ctgan(
        csv_path=CSV_PATH,
        output_path=OUTPUT_PATH,
        epochs=100,
        num_rows=100
    )