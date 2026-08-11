import os
import pandas as pd
from ctgan import CTGAN


def train_ctgan(
    csv_path,
    output_path="generated_csv/synthetic_data.csv",
    epochs=100,
    num_rows=None
):
    print("=" * 60)
    print("CSV SYNTHETIC DATA GENERATION")
    print("=" * 60)

    if not os.path.exists(csv_path):
        print("ERROR: CSV file not found:")
        print(csv_path)
        return

    print("Loading CSV dataset...")

    data = pd.read_csv(csv_path)

    print("Dataset loaded successfully")
    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    if data.empty:
        print("ERROR: CSV dataset is empty.")
        return

    print("-" * 60)
    print("Training CTGAN...")

    model = CTGAN(
        epochs=epochs,
        verbose=True
    )

    model.fit(data)

    print("CTGAN training completed")

    if num_rows is None:
        num_rows = len(data)

    print("-" * 60)
    print("Generating synthetic CSV data...")

    synthetic_data = model.sample(num_rows)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    synthetic_data.to_csv(
        output_path,
        index=False
    )

    print("Synthetic CSV generated successfully")
    print("Generated rows:", len(synthetic_data))
    print("Saved to:", output_path)

    print("=" * 60)