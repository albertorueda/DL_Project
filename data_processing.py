from modules.dataset import (
    df_create,
    add_timestamps,
    split_dataset,
    interp_prev_next_on_rounded,
    INTERPOLATE,
    ROUND_INTERVAL_MIN,
)

# =============================================
# --- LOAD AND PREPROCESS FULL DATASET ---
# =============================================
data_file = "data/aisdk-2025-02-27.csv"
df = df_create(data_file)

# =============================================
# --- OPTIONAL INTERPOLATION STEP ---
# =============================================
if INTERPOLATE:
    print(f"[INFO] Interpolating to {ROUND_INTERVAL_MIN}-minute intervals...")
    df = interp_prev_next_on_rounded(df, n=ROUND_INTERVAL_MIN)
else:
    print(f"[INFO] Adding timestamp rounding columns...")
    df = add_timestamps(df, f'{ROUND_INTERVAL_MIN}min')

# =============================================
# --- TRAIN/VAL/TEST SPLIT ---
# =============================================
train_df, val_df, test_df = split_dataset(df, train_frac=0.7, val_frac=0.15)

print(f"Training set size: {len(train_df)}")
train_df.to_csv("data/train.csv", index=False)

print(f"Validation set size: {len(val_df)}")
val_df.to_csv("data/val.csv", index=False)

print(f"Test set size: {len(test_df)}")
test_df.to_csv("data/test.csv", index=False)