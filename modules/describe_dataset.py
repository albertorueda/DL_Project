"""
describe_dataset.py

Compute descriptive statistics for the cleaned AIS dataset.
This file receives the already–cleaned DataFrame produced in dataset.py
and generates summary metrics useful for understanding the dataset before training.

All metrics here are *descriptive only* (not model-related).
"""



### ================================================================
### --- IMPORTS ---
### ================================================================
# Third-party libraries
import pandas as pd
import numpy as np



# ================================================================
# --- METRIC FUNCTIONS ---
# ================================================================
def compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute high-level dataset statistics:
    - number of rows
    - number of unique ships
    - time span
    - number of features
    """
    stats = {
        "total_rows": len(df),
        "unique_ships": df["MMSI"].nunique(),
        "time_min": df["Timestamp"].min(),
        "time_max": df["Timestamp"].max(),
        "total_timespan_hours": (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() / 3600,
        "num_features": len(df.columns),
    }
    return pd.DataFrame(stats, index=[0])


def compute_ship_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how many ships belong to each ship type (Class A vs Class B).
    Requires that df contains column 'ShipType' or equivalent.
    If the column does not exist, returns an empty DataFrame.
    """
    if "ShipType" not in df.columns:
        return pd.DataFrame({"warning": ["ShipType column missing"]})
    
    ship_counts = df.groupby("ShipType")["MMSI"].nunique().reset_index(name="num_ships")
    row_counts = df.groupby("ShipType").size().reset_index(name="num_rows")
    return ship_counts.merge(row_counts, on="ShipType")


def compute_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics about segments:
    - number of segments per ship
    - average segment length
    - distribution of segment lengths
    If no segmentation was used, return a warning.
    """
    if "Segment" not in df.columns:
        return pd.DataFrame({"warning": ["Segmentation not applied in dataset"]})
    
    segment_lengths = df.groupby(["MMSI", "Segment"]).size().reset_index(name="length")
    
    stats = {
        "total_segments": len(segment_lengths),
        "mean_segment_length": segment_lengths["length"].mean(),
        "median_segment_length": segment_lengths["length"].median(),
        "min_segment_length": segment_lengths["length"].min(),
        "max_segment_length": segment_lengths["length"].max(),
    }
    return pd.DataFrame(stats, index=[0])


def compute_speed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for SOG (Speed Over Ground):
    - mean, median, percentile distribution
    - outlier counts
    """
    sog = df["SOG"].dropna()

    stats = {
        "sog_mean": sog.mean(),
        "sog_median": sog.median(),
        "sog_std": sog.std(),
        "sog_min": sog.min(),
        "sog_max": sog.max(),
        "sog_p90": np.percentile(sog, 90),
        "sog_p95": np.percentile(sog, 95),
        "sog_p99": np.percentile(sog, 99),
        "num_outliers_above_30ms": (sog > 30).sum(),
    }
    return pd.DataFrame(stats, index=[0])


def compute_cog_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive stats for COG (circular quantity).
    Provide circular mean and dispersion.
    """
    if df["COG"].isna().all():
        return pd.DataFrame({"warning": ["No valid COG values"]})

    cog_rad = np.deg2rad(df["COG"].dropna())
    sin_mean = np.mean(np.sin(cog_rad))
    cos_mean = np.mean(np.cos(cog_rad))
    circular_mean = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360

    R = np.sqrt(sin_mean**2 + cos_mean**2)
    circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.nan

    stats = {
        "cog_circular_mean_deg": circular_mean,
        "cog_circular_std": circular_std,
        "cog_min": df["COG"].min(),
        "cog_max": df["COG"].max(),
    }
    return pd.DataFrame(stats, index=[0])


def compute_sampling_gap_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time gap statistics between consecutive AIS messages per ship.
    Helps evaluate irregularity of sampling.
    """
    df_sorted = df.sort_values(["MMSI", "Timestamp"])
    df_sorted["delta_sec"] = df_sorted.groupby("MMSI")["Timestamp"].diff().dt.total_seconds()

    gaps = df_sorted["delta_sec"].dropna()

    stats = {
        "gap_mean_sec": gaps.mean(),
        "gap_median_sec": gaps.median(),
        "gap_90th_sec": np.percentile(gaps, 90),
        "gap_99th_sec": np.percentile(gaps, 99),
        "num_gaps_over_30min": (gaps > 1800).sum(),
    }
    return pd.DataFrame(stats, index=[0])



# ================================================================
# --- MAIN ENTRY POINT ---
# ================================================================
def compute_all_metrics(df: pd.DataFrame) -> dict:
    """
    Compute and return all dataset metrics in a dictionary of DataFrames.
    """
    return {
        "basic_stats": compute_basic_stats(df),
        "ship_type_distribution": compute_ship_type_distribution(df),
        "segment_stats": compute_segment_stats(df),
        "speed_stats": compute_speed_stats(df),
        "cog_stats": compute_cog_stats(df),
        "sampling_gap_stats": compute_sampling_gap_stats(df),
    }


def save_all_metrics(df: pd.DataFrame, output_path: str):
    """
    Save all metrics as CSV files inside a folder.
    """
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    metrics = compute_all_metrics(df)

    for name, table in metrics.items():
        table.to_csv(f"{output_path}/{name}.csv", index=False)

    print(f"[INFO] All metrics saved to folder: {output_path}")