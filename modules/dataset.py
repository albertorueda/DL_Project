import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

def df_create(file_path: str) -> pd.DataFrame:
    """
    Create a cleaned DataFrame from raw AIS data.
    
    Args:
        file_path (str): Path to the CSV file containing raw AIS data.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with relevant AIS data.
    """
    
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
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtypes)

    # Remove errors
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
            df["Longitude"] <= east)]
    
    
    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    df = df.drop_duplicates(["Timestamp", "MMSI", ], keep="first")

    #PABLO --> Should we add this filter, or thats what the Deep Learning model is for? 
    #def track_filter(g):
    #    len_filt = len(g) > 256  # Min required length of track/segment
    #    sog_filt = 1 <= g["SOG"].max() <= 50  # Remove stationary tracks/segments
    #    time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60  # Min required timespan
    #    return len_filt and sog_filt and time_filt

    ## Track filtering
    #df = df.groupby("MMSI").filter(track_filter)
    #df = df.sort_values(['MMSI', 'Timestamp'])

    ## Divide track into segments based on timegap
    #df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
    #    lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum())  # Max allowed timegap

    ## Segment filtering
    #df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    #df = df.reset_index(drop=True)
    
    df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)

    #Now we have it in m/s
    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]
    return df


def add_timestamps(df: pd.DataFrame, round_interval: str) -> pd.DataFrame:
    """
    Add rounded timestamp columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        round_interval (str): Rounding interval (e.g., '10min').
    
    Returns:
        pd.DataFrame: DataFrame with added rounded timestamp columns.        
    """

    # 1) Columns of rounded timestamps
    df["timestamp_v2"] = pd.to_datetime(df["Timestamp"], utc=False, errors="coerce").dt.floor(round_interval)
    df["timestamp_v3"] = pd.to_datetime(df["Timestamp"], utc=False, errors="coerce").dt.ceil(round_interval)

    # Order by MMSI and actual time
    df = df.sort_values(["MMSI", "Timestamp"])

    # 2) Mark first/last within each group (MMSI + timestamp_v2)
    pos_v2 = df.groupby(["MMSI", "timestamp_v2"]).cumcount() + 1
    cnt_v2 = df.groupby(["MMSI", "timestamp_v2"])["Timestamp"].transform("size")
    keep_v2 = (pos_v2 == 1)

    # 3) Mark first/last within each group (MMSI + timestamp_v3)
    pos_v3 = df.groupby(["MMSI", "timestamp_v3"]).cumcount() + 1
    cnt_v3 = df.groupby(["MMSI", "timestamp_v3"])["Timestamp"].transform("size")
    keep_v3 = (pos_v3 == cnt_v3)

    # 4) Keep rows that are first/last in v2 or in v3
    mask = keep_v2.fillna(False) | keep_v3.fillna(False)
    df3 = df[mask].copy().reset_index(drop=True).drop(columns=["timestamp_v2", "timestamp_v3"])
    
    return df3


def split_dataset(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): The DataFrame to split.
        train_frac (float): Fraction of data to use for training.
        val_frac (float): Fraction of data to use for validation.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test DataFrames.
    """

    ship_number = df["MMSI"].nunique()

    train_ships = random.sample(list(df["MMSI"].unique()), int(ship_number * train_frac))
    val_ships = random.sample(list(set(df["MMSI"].unique()) - set(train_ships)), int(ship_number * val_frac))
    test_ships = list(set(df["MMSI"].unique()) - set(train_ships) - set(val_ships))

    df_train = df[df["MMSI"].isin(train_ships)].reset_index(drop=True)
    df_val = df[df["MMSI"].isin(val_ships)].reset_index(drop=True)
    df_test = df[df["MMSI"].isin(test_ships)].reset_index(drop=True)

    return df_train, df_val, df_test


def interp_linear(a, b, r) -> np.ndarray:
    """
    Linear interpolation between a and b with ratio r.
    
    Args:
        a: Starting values.
        b: Ending values.
        r: Ratio for interpolation.
        
    Returns:
        Interpolated values.
    """
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(r, errors="coerce").to_numpy(dtype=float)
    y = a + r * (b - a)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(r)
    y[~ok] = np.nan
    return y


def interp_circular_deg(a, b, r) -> np.ndarray:
    """
    Circular interpolation between angles a and b (in degrees) with ratio r.
    
    Args:
        a: Starting angles in degrees.
        b: Ending angles in degrees.
        r: Ratio for interpolation.
    
    Returns:
        Interpolated angles in degrees.
    """
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(r, errors="coerce").to_numpy(dtype=float)
    delta = ( (b - a + 180.0) % 360.0 ) - 180.0
    y = a + r * delta
    y = (y + 360.0) % 360.0
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(r)
    y[~ok] = np.nan
    return y


def interp_prev_next_on_rounded(
    df, n=5,
    value_cols=("Latitude", "Longitude", "SOG", "COG"),
    circular_cols=("COG",),                
    group_cols=("MMSI",), ts_col="Timestamp",
    drop_timezone=True,                    
) -> pd.DataFrame:
    """
    Interpolation based on previous and next values on rounded timestamps.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        n (int): Rounding interval in minutes.
        value_cols (tuple): Columns to interpolate.
        circular_cols (tuple): Columns that are circular (angles in degrees).
        group_cols (tuple): Columns to group by.
        ts_col (str): Timestamp column name.
        drop_timezone (bool): Whether to drop timezone info from timestamps.
        
    Returns:
        pd.DataFrame: Interpolated DataFrame.
    """
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d = d.sort_values(list(group_cols) + [ts_col])

    # 1) floor/ceil to n minutes
    d["timestamp_v2"] = d[ts_col].dt.floor(f"{n}min")
    d["timestamp_v3"] = d[ts_col].dt.ceil(f"{n}min")

    out = []
    for keys, g in d.groupby(list(group_cols), dropna=False, sort=False):
        g = g.dropna(subset=[ts_col]).sort_values(ts_col)
        if g.empty:
            continue

        # 2) targets = union(floor, ceil) within the observed range
        targets = pd.Series(
            pd.concat([g["timestamp_v2"], g["timestamp_v3"]], ignore_index=True)
        ).dropna().unique()
        tmin, tmax = g[ts_col].min(), g[ts_col].max()
        targets = [t for t in targets if (tmin <= t <= tmax)]
        if not targets:
            continue

        tgt = pd.DataFrame({ts_col: sorted(pd.to_datetime(targets))})
        if isinstance(keys, tuple):
            for c, v in zip(group_cols, keys): tgt[c] = v
        else:
            tgt[group_cols[0]] = keys

        base = g[list(group_cols) + [ts_col] + list(value_cols)].sort_values(ts_col)

        # 3) previous neighbor
        prev = pd.merge_asof(
            tgt.sort_values(ts_col),
            base.assign(_t_prev=base[ts_col]),
            on=ts_col, by=list(group_cols), direction="backward"
        ).rename(columns={c: f"{c}_prev" for c in value_cols})

        # next neighbor
        nxt = pd.merge_asof(
            tgt.sort_values(ts_col),
            base.assign(_t_next=base[ts_col]),
            on=ts_col, by=list(group_cols), direction="forward"
        ).rename(columns={c: f"{c}_next" for c in value_cols})

        m = prev.merge(
            nxt[list(group_cols) + [ts_col, "_t_next"] + [f"{c}_next" for c in value_cols]],
            on=list(group_cols) + [ts_col], how="inner"
        )

        # ratio = (t - t_prev) / (t_next - t_prev)
        num = (m[ts_col] - m["_t_prev"]).dt.total_seconds()
        den = (m["_t_next"] - m["_t_prev"]).dt.total_seconds()
        r = (num / den).where(np.isfinite(num) & np.isfinite(den) & (den != 0))

        # 4) Interpolation: linear vs circular
        for c in value_cols:
            if c in circular_cols:
                m[c] = interp_circular_deg(m[f"{c}_prev"], m[f"{c}_next"], r)
            else:
                m[c] = interp_linear(m[f"{c}_prev"], m[f"{c}_next"], r)

        out.append(m[list(group_cols) + [ts_col] + list(value_cols)])
        
    if not out:
        res = pd.DataFrame(columns=list(group_cols) + [ts_col] + list(value_cols))
    else:
        res = pd.concat(out, ignore_index=True).sort_values(list(group_cols) + [ts_col])
        res = res.drop_duplicates(subset=list(group_cols) + [ts_col], keep="first")

    # 5) remove +00:00 (make naive) if requested
    if drop_timezone and not res.empty:
        res[ts_col] = res[ts_col].dt.tz_localize(None)

    return res


class AISDataset(Dataset):
    def __init__(self, dataset_path: str, seq_input_length: int = 6, seq_output_length: int = 1):
        # We only support seq_output_length of 1 (autoregressive) or equal to seq_input_length (same window size input-output) for now
        if (seq_input_length != seq_output_length and seq_output_length != 1):
            raise ValueError("Either seq_input_length must be equal to seq_output_length or seq_output_length must be 1.")
        
        self.dataframe = pd.read_csv(dataset_path)
        self.seq_input_length = seq_input_length
        self.seq_output_length = seq_output_length
        self.valid_idxs = []
        
        # Precompute valid indices
        unique_mmsis = self.dataframe['MMSI'].unique()
        for mmsi in unique_mmsis:
            # Get all rows for this MMSI execpt the last seq_input_length + seq_output_length - 1 rows
            mmsi_data = self.dataframe[self.dataframe['MMSI'] == mmsi]
            # Add the indexes of valid starting points for sequences
            for start_idx in range(len(mmsi_data) - (self.seq_input_length + self.seq_output_length) + 1):
                real_idx = mmsi_data.index[start_idx]
                self.valid_idxs.append(real_idx)
                
        # Normalization parameters
        #self.lat_min = self.dataframe['Latitude'].min()
        #self.lat_max = self.dataframe['Latitude'].max()
        #self.lon_min = self.dataframe['Longitude'].min()
        #self.lon_max = self.dataframe['Longitude'].max()
        #
        #self.dataframe['Latitude'] = (self.dataframe['Latitude'] - self.lat_min) / (self.lat_max - self.lat_min)
        #self.dataframe['Longitude'] = (self.dataframe['Longitude'] - self.lon_min) / (self.lon_max - self.lon_min)

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx):
        real_idx = self.valid_idxs[idx]
        x = self.dataframe.iloc[real_idx: real_idx + self.seq_input_length][['Latitude', 'Longitude', 'SOG', 'COG']].to_numpy(dtype=float)
        y = self.dataframe.iloc[real_idx + self.seq_input_length: real_idx + self.seq_input_length + self.seq_output_length][['Latitude', 'Longitude']].to_numpy(dtype=float)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


#