"""
ais_dataset.py

Script to generate the train/validation/test splits from the fully cleaned and preprocessed AIS dataset.

It loads the full dataset from a CSV, splits it by MMSI into train/val/test sets,
and exports them as separate files. This script is designed to be run after preprocessing.
"""


# =============================================
# --- IMPORTS ---
# =============================================
from modules.ais_preprocessing import (
    df_create,
    split_dataset
)
from modules.describe_dataset import (
    save_all_metrics
)
import glob, os


# =============================================
# --- LOAD AND PREPROCESS AIS DATASET ---
# =============================================
print("Scanning raw data folder for AIS CSV files...")
raw_files = glob.glob("raw data/*.csv")

if not raw_files:
    raise FileNotFoundError("No CSV files found in raw_data/")

for file_path in raw_files:
    base = os.path.basename(file_path)          # e.g. 2022-01-15.csv
    name, _ = os.path.splitext(base)            # e.g. 2022-01-15
    print(f"\n=== Processing file: {base} ===")

    df = df_create(file_path)

    # Save metrics per file
    metrics_dir = f"metrics/{name}"
    os.makedirs(metrics_dir, exist_ok=True)
    save_all_metrics(df, metrics_dir)

    # Split dataset
    print("Splitting into train/val/test...")
    train_df, val_df, test_df = split_dataset(df, train_frac=0.7, val_frac=0.15)

    # Create folder
    os.makedirs("datasplits", exist_ok=True)

    train_df.to_csv(f"datasplits/train_{name}.csv", index=False)
    val_df.to_csv(f"datasplits/val_{name}.csv", index=False)
    test_df.to_csv(f"datasplits/test_{name}.csv", index=False)

    print(f"Saved: train_{name}.csv, val_{name}.csv, test_{name}.csv")