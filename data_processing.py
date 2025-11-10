from modules.dataset import df_create, add_timestamps, split_dataset, interp_prev_next_on_rounded

data_file = "data/aisdk-2025-02-27.csv"

df = df_create(data_file)
df = add_timestamps(df, '10min')
#df = interp_prev_next_on_rounded(df, 10)

train_df, val_df, test_df = split_dataset(df, train_frac=0.7, val_frac=0.15)

print(f"Training set size: {len(train_df)}")
train_df.to_csv("data/train_set.csv", index=False)

print(f"Validation set size: {len(val_df)}")
val_df.to_csv("data/val_set.csv", index=False)

print(f"Test set size: {len(test_df)}")
test_df.to_csv("data/test_set.csv", index=False)
