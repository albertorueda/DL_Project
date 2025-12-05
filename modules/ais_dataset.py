"""
ais_dataset.py

This script automates the generation of train/validation/test splits from cleaned AIS data.
It iterates over all raw CSV files in the `raw data/` directory, applies the preprocessing pipeline,
and exports the splits to the `datasplits/` folder.

Expected file structure:
- Input: raw data/*.csv
- Output: datasplits/train/train_*.csv, datasplits/val/val_*.csv, datasplits/test/test_*.csv

Run this after `ais_preprocessing` has been finalized.
"""

# =============================================
# --- IMPORTS ---
# =============================================

from modules.ais_preprocessing import (
    df_create,
    split_dataset
)
import glob
import os

# =============================================
# --- LOAD AND PREPROCESS AIS DATASET ---
# =============================================

print("Scanning raw data folder for AIS CSV files...")
raw_files = glob.glob("raw data/*.csv")

if not raw_files:
    raise FileNotFoundError("No CSV files found in raw_data/")

for file_path in raw_files:
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    print(f"\n=== Processing file: {base} ===")

    # --- Preprocess file ---
    df = df_create(file_path)

    # --- Split dataset ---
    print("Splitting into train/val/test...")
    train_df, val_df, test_df = split_dataset(df, train_frac=0.7, val_frac=0.15)

    # --- Export CSVs ---
    os.makedirs("datasplits", exist_ok=True)  # Ensure datasplits directory exists
    train_df.to_csv(f"datasplits/train/train_{name}.csv", index=False)
    val_df.to_csv(f"datasplits/val/val_{name}.csv", index=False)
    test_df.to_csv(f"datasplits/test/test_{name}.csv", index=False)

    print(f"Saved: train_{name}.csv, val_{name}.csv, test_{name}.csv")