"""
ais_preprocessing.py

This module handles AIS data loading, cleaning, interpolation, normalization,
and dataset splitting for the vessel trajectory prediction project.

It implements:
- Robust AIS cleaning (bounding box, MMSI validation, timestamp parsing,
  duplicate removal, zero–coordinate filtering, speed outlier removal).
- Track-level filtering and optional segmentation to ensure high–quality
  vessel trajectories.
- Linear and circular interpolation using pandas.resample to create
  fixed-interval time series.
- Optional normalization for Latitude/Longitude.
- Final dataset preparation and splitting by vessel (MMSI).

All steps include comments intended for pedagogical clarity in the context
of the DTU 02456 Deep Learning course project.
"""



### ================================================================
### --- IMPORTS ---
### ================================================================
# Standard libraries
import random

# Third-party libraries
import pandas as pd
import numpy as np



### ================================================================
### --- DATAFRAME CREATION ---
### ================================================================
def df_create(file_path: str) -> pd.DataFrame:
    """
    Create a cleaned DataFrame from raw AIS data.
    """

    ### ================================================================
    ### --- FLAGS ---
    ### ================================================================
    REMOVE_OUT_OF_BOUNDS = True
    ENABLE_TYPE_FILTER = True
    MMSI_VALIDATION = True
    MMSI_MID_RANGE_VALIDATION = True
    REMOVE_INVALID_TIMESTAMPS = True
    REMOVE_DUPLICATES = True
    REMOVE_ZERO_COORDS = True
    REMOVE_SOG_COG_NAN_IF_NO_INTERPOLATION = True
    REMOVE_HIGH_SOG = True
    REMOVE_MIN_VALID_YEAR = 2015

    ENABLE_TRACK_FILTER = True
    ENABLE_SEGMENTATION = True

    ENABLE_INTERPOLATION = True
    ENABLE_NORMALIZATION = True
    
    ### ================================================================
    ### --- CONSTANTS ---
    ### ================================================================
    KNOTS_TO_MS = 0.514444  # Conversion factor from knots to meters per second

    BBOX = [60, 0, 50, 20]
    VALID_TYPES = ["Class A", "Class B"]
    MMSI_STANDARD_LENGTH = 9
    VALID_MID_RANGE = (200, 775)
    MAX_SOG_THRESHOLD = 30
    MIN_VALID_YEAR = 2015
    MIN_TRACK_LENGTH = 256
    MAX_SOG_FOR_TRACK = 50

    MIN_TRACK_TIMESPAN_SECONDS = 3600
    SEGMENT_GAP_SECONDS = 900
    MIN_SEGMENT_LENGTH = 10

    ROUND_INTERVAL_MIN = 10

    ### ================================================================
    ### STEP 1: READ CSV
    ### ================================================================
    # Define expected data types to optimize memory usage and ensure consistency
    dtypes = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
    }
    usecols = list(dtypes.keys())
    # Load raw AIS data from CSV file with specified columns and types
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtypes)
    initial_rows = len(df)
    print(f"[INFO] Loaded raw CSV: {initial_rows} rows.")

    ### --- CHECK FOR MISSING EXPECTED COLUMNS ---
    # Validate presence of critical columns to avoid downstream errors
    expected_columns = {"MMSI", "Timestamp", "Latitude", "Longitude", "SOG", "COG"}
    missing_cols = expected_columns - set(df.columns)
    if missing_cols:
        print(f"[WARNING] The following expected columns are missing from the dataframe: {missing_cols}")

    ### --- CONVERT SOG FROM KNOTS TO M/S IMMEDIATELY AFTER READING ---
    # Convert SOG from knots to m/s for consistency in physical units
    df["SOG"] = KNOTS_TO_MS * df["SOG"]  # All further processing will be done in meters per second

    ### ================================================================
    ### STEP 2: BASIC CLEANING
    ### ================================================================
    ### --- REMOVE OUT-OF-BOUNDS POSITIONS ---
    # Filter out AIS points outside the defined geographic bounding box to focus on the area of interest
    if REMOVE_OUT_OF_BOUNDS:
        north, west, south, east = BBOX
        initial_rows = len(df)
        df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (df["Longitude"] <= east)]
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows outside bounding box.")

    ### --- FILTER BY SHIP TYPE AND MMSI VALIDATION ---
    # Keep only relevant ship types to reduce noise and ensure data quality
    if ENABLE_TYPE_FILTER:
        initial_rows = len(df)
        df = df[df["Type of mobile"].isin(VALID_TYPES)].drop(columns=["Type of mobile"])
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows not matching valid vessel types.")

    ### --- MMSI FORMAT VALIDATION ---
    # Validate MMSI length to comply with standardized maritime identifiers
    if MMSI_VALIDATION and MMSI_STANDARD_LENGTH:
        initial_rows = len(df)
        df = df[df["MMSI"].str.len() == MMSI_STANDARD_LENGTH]  # Adhere to MMSI format
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows failing MMSI length validation.")
    
    ### --- MMSI MID RANGE VALIDATION ---
    # Filter by MID range to ensure vessels are registered in expected regions
    if MMSI_MID_RANGE_VALIDATION and VALID_MID_RANGE is not None:
        initial_rows = len(df)
        df = df[df["MMSI"].str[:3].astype(int).between(VALID_MID_RANGE[0], VALID_MID_RANGE[1])]  # Adhere to MID standard
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows failing MMSI MID range validation.")

    ### --- TIMESTAMP PARSING ---
    # Rename timestamp column and convert to datetime for temporal analysis
    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    ### --- TIMESTAMP VALIDATION ---
    # Remove rows with invalid timestamps to maintain temporal integrity
    if REMOVE_INVALID_TIMESTAMPS:
        initial_rows = len(df)
        invalid_ts = df["Timestamp"].isna().sum()
        if invalid_ts > 0:
            print(f"[WARNING] {invalid_ts} / {initial_rows} rows have invalid timestamps and will be removed.")
        df = df.dropna(subset=["Timestamp"])
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows with invalid timestamps.")

    ### --- REMOVE DUPLICATE ENTRIES ---
    # Remove duplicate AIS messages to avoid bias and redundancy
    if REMOVE_DUPLICATES:
        initial_rows = len(df)
        df = df.drop_duplicates(["Timestamp", "MMSI", ], keep="first")
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} duplicate rows.")

    ### --- REMOVE INVALID COORDS ---
    # Exclude points with zero coordinates which are likely erroneous
    if REMOVE_ZERO_COORDS:
        initial_rows = len(df)
        df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)]
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows with zero coordinates.")

    ### --- REMOVE NaN SOG/COG IF NO INTERPOLATION ---
    # Drop rows with missing speed or course if interpolation is not enabled to keep data consistent
    if REMOVE_SOG_COG_NAN_IF_NO_INTERPOLATION and not ENABLE_INTERPOLATION:
        initial_rows = len(df)
        df = df.dropna(subset=["SOG", "COG"])
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows with NaN SOG/COG (no interpolation).")

    ### --- REMOVE HIGH SOG VALUES ---
    # Filter out unrealistic high speeds to remove noise and outliers
    if REMOVE_HIGH_SOG and MAX_SOG_THRESHOLD is not None:
        initial_rows = len(df)
        df = df[df["SOG"] < MAX_SOG_THRESHOLD]
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows with SOG >= {MAX_SOG_THRESHOLD} m/s.")

    ### --- REMOVE DATA BEFORE MIN VALID YEAR ---
    # Focus analysis on recent data by excluding older records
    if REMOVE_MIN_VALID_YEAR and MIN_VALID_YEAR is not None:
        initial_rows = len(df)
        df = df[df["Timestamp"].dt.year >= MIN_VALID_YEAR]
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows before year {MIN_VALID_YEAR}.")

    ### ================================================================
    ### STEP 3: TRACK FILTERING AND SEGMENTATION
    ### ================================================================
    def track_filter(g):
        """
        Filter tracks based on length, speed, and timespan criteria.
        Ensures only meaningful vessel trajectories are kept for modeling.
        """
        len_filt = len(g) > MIN_TRACK_LENGTH  # Min required length of track/segment
        sog_filt = 1 <= g["SOG"].max() <= MAX_SOG_FOR_TRACK  # Remove stationary or outlier segments
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= MIN_TRACK_TIMESPAN_SECONDS
        return len_filt and sog_filt and time_filt

    # Filter out tracks that do not meet minimum criteria to improve model training quality
    if ENABLE_TRACK_FILTER:
        initial_rows = len(df)
        df = df.groupby("MMSI").filter(track_filter)
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows by track filtering.")

    if ENABLE_SEGMENTATION:
        # Segment tracks based on time gaps to isolate continuous vessel movements
        df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
            lambda x: (x.diff().dt.total_seconds().fillna(0) >= SEGMENT_GAP_SECONDS).cumsum()
        )
        initial_rows = len(df)
        df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows by segment track filtering.")

        initial_rows = len(df)
        # Remove short segments after segmentation to keep only substantial trajectories
        df = df.groupby(["MMSI", "Segment"]).filter(lambda g: len(g) >= MIN_SEGMENT_LENGTH)
        removed = initial_rows - len(df)
        print(f"[INFO] Removed {removed} rows from segments shorter than {MIN_SEGMENT_LENGTH} rows.")

        df = df.reset_index(drop=True)
    else:
        df['Segment'] = 0  # Default segment assignment when segmentation is disabled

    ### ================================================================
    ### STEP 4: INTERPOLATION AND NORMALIZATION
    ### ================================================================
    if ENABLE_INTERPOLATION:
        print(f"[INFO] Interpolating dataset at {ROUND_INTERVAL_MIN}-minute intervals using pandas.resample...")
        # Set timestamp as index for time-based resampling
        df = df.set_index("Timestamp")

        # ================================================================
        # --- LINEAR INTERPOLATION FOR LINEAR VARIABLES ---
        # ================================================================
        # Columns that can be interpolated linearly
        linear_cols = ["Latitude", "Longitude", "SOG"]
        
        def interpolate_linear(df, interval="10min", group_col="MMSI", time_col="Timestamp", method="linear"):
            """
            Interpolate data at fixed time intervals using pandas' resample function.
            This ensures consistent temporal resolution for AIS trajectories.
            """
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col])
            df = df.sort_values([group_col, time_col])

            # Resample and interpolate to fill missing data points at regular intervals
            resampled_df = (
                df.groupby(group_col, group_keys=False)
                .apply(lambda group: group.set_index(time_col)
                                        .resample(interval)
                                        .mean()
                                        .interpolate(method=method)
                                        .reset_index())
            )

            return resampled_df

        # Perform linear interpolation for position and speed
        df_linear = interpolate_linear(
            df[linear_cols + ["MMSI"]], 
            interval=f"{ROUND_INTERVAL_MIN}min",
            group_col="MMSI",
            time_col="Timestamp",
            method="linear"
        )
        print(f"[INFO] Linear interpolated rows: {len(df_linear)}")

        # ================================================================
        # --- CIRCULAR INTERPOLATION FOR ANGULAR VARIABLES ---
        # ================================================================
        # Columns that represent angles and require special interpolation to handle wrap-around
        circular_cols = ["COG"]  # Columns measured in degrees on a circular domain

        def interpolate_circular(group):
            """
            Perform circular interpolation for angular columns like COG.
            This handles angle wrap-around correctly by interpolating sine and cosine components.
            """
            # Work on a copy to avoid modifying original group during iteration
            result = group.copy()

            for col in circular_cols:
                # 1. Fallback interpolation (pad) to avoid NaNs before trig transform
                raw = result[col].copy().interpolate(method="pad")

                # 2. Convert to radians
                rad = np.deg2rad(raw)

                # 3. Interpolate sine and cosine independently
                sin_interp = np.sin(rad).interpolate(method="linear")
                cos_interp = np.cos(rad).interpolate(method="linear")

                # 4. Reconstruct angle using arctan2
                angle_rad = np.arctan2(sin_interp, cos_interp)

                # 5. Convert back to degrees and wrap to [0, 360)
                result[col] = np.rad2deg(angle_rad) % 360

            # Return only circular columns
            return result[circular_cols]

        # Apply circular interpolation per MMSI to handle angular data correctly
        df_cog = df.groupby("MMSI").apply(interpolate_circular)
        print(f"[INFO] Circular interpolated rows: {len(df_cog)}")

        # ================================================================
        # --- COMBINE INTERPOLATIONS ---
        # ================================================================
        # Merge linear and circular interpolated data into a single DataFrame
        df_interp = pd.concat([df_linear, df_cog], axis=1).reset_index()
        df = df_interp
        # Drop any remaining rows with missing critical values
        initial_rows = len(df)
        df = df.dropna(subset=["Latitude", "Longitude", "SOG", "COG"])
        removed = initial_rows - len(df)
        print(f"[INFO] Interpolated dataset now has {len(df)} rows (removed {removed} rows with missing critical values).")

    if ENABLE_NORMALIZATION:
        # Normalize Latitude and Longitude to [0,1] to facilitate machine learning model convergence
        print("[INFO] Normalizing Latitude and Longitude columns to [0,1] range.")
        df["Latitude"] = (df["Latitude"] - df["Latitude"].min()) / (df["Latitude"].max() - df["Latitude"].min())
        df["Longitude"] = (df["Longitude"] - df["Longitude"].min()) / (df["Longitude"].max() - df["Longitude"].min())

    # Sort data by MMSI and Timestamp to maintain chronological order for each vessel
    df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)

    ### ================================================================
    ### STEP 4: SUMMARY AND FINALIZATION
    ### ================================================================
    # Print summary statistics about the cleaned dataset
    print(f"[INFO] Final dataset stats:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Unique MMSIs: {df['MMSI'].nunique()}")
    print(f"  - Time span: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

    return df



### ================================================================
### --- DATA SPLITTING FUNCTION ---
### ================================================================
def split_dataset(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training, validation, and test sets.
    """

    # Determine number of unique vessels to split by vessel rather than by row
    ship_number = df["MMSI"].nunique()

    # Randomly sample vessels for training set
    train_ships = random.sample(list(df["MMSI"].unique()), int(ship_number * train_frac))
    # Sample vessels for validation set excluding training vessels
    val_ships = random.sample(list(set(df["MMSI"].unique()) - set(train_ships)), int(ship_number * val_frac))
    # Remaining vessels assigned to test set
    test_ships = list(set(df["MMSI"].unique()) - set(train_ships) - set(val_ships))

    # Create DataFrames for each split based on vessel membership
    df_train = df[df["MMSI"].isin(train_ships)].reset_index(drop=True)
    df_val = df[df["MMSI"].isin(val_ships)].reset_index(drop=True)
    df_test = df[df["MMSI"].isin(test_ships)].reset_index(drop=True)

    return df_train, df_val, df_test