"""
ais_dataset.py

This script automates the preprocessing and splitting of AIS datasets.
It iterates over each CSV file in the raw_data/ directory, applies cleaning,
interpolation, and filtering (via `df_create`), then splits into train/val/test
sets using vessel-based grouping.

Output CSVs are saved into datasplits/train/, datasplits/val/, and datasplits/test/
with filenames corresponding to each input day.
"""

### ================================================================
### --- IMPORTS ---
### ================================================================

from modules.ais_preprocessing import (
    df_create,
    split_dataset
)
import glob
import os

### ================================================================
### --- LOAD RAW AIS DATA FILES ---
### ================================================================

print("Scanning raw_data folder for AIS CSV files...")
# Find all CSV files in the raw_data folder
raw_files = glob.glob(os.path.join("raw_data", "*.csv"))

if not raw_files:
    raise FileNotFoundError("No CSV files found in raw_data/")

for file_path in raw_files:
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    print(f"\n=== Processing file: {base} ===")

    ### ================================================================
    ### --- PREPROCESS AIS DATA ---
    ### ================================================================
    # Load and clean the AIS data using the preprocessing pipeline
    df = df_create(file_path)

    ### ================================================================
    ### --- SPLIT DATASET INTO TRAIN/VAL/TEST ---
    ### ================================================================
    print("Splitting into train/val/test...")
    # Split dataset with vessel-based grouping to avoid data leakage
    train_df, val_df, test_df = split_dataset(df, train_frac=0.7, val_frac=0.15)

    ### ================================================================
    ### --- EXPORT SPLIT DATASETS TO CSV ---
    ### ================================================================
    # Ensure output directories exist
    os.makedirs(os.path.join("datasplits", "train"), exist_ok=True)
    os.makedirs(os.path.join("datasplits", "val"), exist_ok=True)
    os.makedirs(os.path.join("datasplits", "test"), exist_ok=True)

    # Save the split datasets to their respective folders
    train_df.to_csv(os.path.join("datasplits", "train", f"train_{name}.csv"), index=False)
    val_df.to_csv(os.path.join("datasplits", "val", f"val_{name}.csv"), index=False)
    test_df.to_csv(os.path.join("datasplits", "test", f"test_{name}.csv"), index=False)

    print(f"Saved: train_{name}.csv, val_{name}.csv, test_{name}.csv")