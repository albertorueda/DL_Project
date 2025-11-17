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


# =============================================
# --- LOAD AND PREPROCESS AIS DATASET ---
# =============================================
print("Loading and preprocessing AIS dataset...")
data_file = "data/aisdk-2025-02-27.csv"
df = df_create(data_file)
print(df.head())


# =============================================
# --- DATASET METRICS ---
# =============================================
print("Calculating and saving dataset metrics...")
save_all_metrics(df, "metrics")


# =============================================
# --- TRAIN/VAL/TEST SPLIT ---
# =============================================
print("Splitting dataset into train/val/test sets...")
train_df, val_df, test_df = split_dataset(df, train_frac=0.7, val_frac=0.15)

print(f"Training set size: {len(train_df)}")
train_df.to_csv("datasplits/train.csv", index=False)
print("Sample from training set:")
print(train_df.head())

print(f"Validation set size: {len(val_df)}")
val_df.to_csv("datasplits/val.csv", index=False)
print("Sample from validation set:")
print(val_df.head())

print(f"Test set size: {len(test_df)}")
test_df.to_csv("datasplits/test.csv", index=False)
print("Sample from test set:")
print(test_df.head())