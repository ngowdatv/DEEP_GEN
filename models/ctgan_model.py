import os
import pandas as pd
from ctgan import CTGAN


def train_ctgan(
    csv_path,
    output_path="generated_csv/synthetic_data.csv",
    epochs=100,
    num_rows=100
):

    print("=" * 60)
    print("CTGAN CSV GENERATION")
    print("=" * 60)

    print("CSV Dataset:")
    print(csv_path)

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not os.path.exists(csv_path):

        print("ERROR: CSV file not found.")
        print(csv_path)

        return False

    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------

    print("-" * 60)
    print("LOADING CSV DATASET")
    print("-" * 60)

    data = pd.read_csv(csv_path)

    print("CSV loaded successfully.")
    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    print("Columns:")
    print(list(data.columns))

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if data.empty:

        print("ERROR: CSV dataset is empty.")

        return False

    # --------------------------------------------------------
    # IDENTIFY CATEGORICAL COLUMNS
    # --------------------------------------------------------

    discrete_columns = []

    for column in data.columns:

        if (
            data[column].dtype == "object"
            or str(data[column].dtype).startswith("category")
        ):

            discrete_columns.append(column)

    print("-" * 60)
    print("Categorical Columns:")
    print(discrete_columns)

    # --------------------------------------------------------
    # CREATE CTGAN
    # --------------------------------------------------------

    print("-" * 60)
    print("BUILDING CTGAN MODEL")
    print("-" * 60)

    model = CTGAN(
        epochs=epochs,
        verbose=True
    )

    # --------------------------------------------------------
    # TRAIN CTGAN
    # --------------------------------------------------------

    print("-" * 60)
    print("STARTING CTGAN TRAINING")
    print("-" * 60)

    model.fit(
        data,
        discrete_columns=discrete_columns
    )

    print("-" * 60)
    print("CTGAN TRAINING COMPLETED")
    print("-" * 60)

    # --------------------------------------------------------
    # GENERATE SYNTHETIC DATA
    # --------------------------------------------------------

    print("-" * 60)
    print("GENERATING SYNTHETIC CSV DATA")
    print("-" * 60)

    synthetic_data = model.sample(
        num_rows
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # --------------------------------------------------------
    # SAVE SYNTHETIC DATA
    # --------------------------------------------------------

    synthetic_data.to_csv(
        output_path,
        index=False
    )

    print("Synthetic CSV generated successfully.")

    print(
        "Generated rows:",
        len(synthetic_data)
    )

    print("Saved to:")
    print(output_path)

    print("=" * 60)
    print("CTGAN PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return True


if __name__ == "__main__":

    train_ctgan(
        "dataset/csv/data.csv",
        "generated_csv/synthetic_data.csv",
        epochs=10,
        num_rows=20
    )