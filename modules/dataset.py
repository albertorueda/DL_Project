"""
AISDataset: PyTorch Dataset for AIS trajectory sequences.

This dataset supports sequence-to-sequence learning with configurable input and output sequence lengths.
It normalizes Latitude, Longitude, and SOG, and transforms COG into sin and cos components.

If provided, normalization statistics from a training set can be reused for validation/test sets.
"""

### ================================================================
### --- IMPORTS ---
### ================================================================
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

### ================================================================
### --- AISDataset CLASS ---
### ================================================================
class AISDataset(Dataset):
    """
    Custom PyTorch Dataset for AIS (Automatic Identification System) trajectory data.

    Args:
        dataset_path (str or pd.DataFrame): Path to CSV or DataFrame containing AIS data.
        seq_input_length (int): Number of time steps in the input sequence.
        seq_output_length (int): Number of time steps in the output sequence.
        stats (tuple, optional): Tuple of (lat_min, lat_max, lon_min, lon_max, sog_max) to apply consistent normalization.
    """
    def __init__(self, dataset_path: str, seq_input_length: int = 6, seq_output_length: int = 1, stats=None):
        # We only support seq_output_length of 1 (autoregressive) or equal to seq_input_length (same window size input-output) for now
        if (seq_input_length != seq_output_length and seq_output_length != 1):
            raise ValueError("Either seq_input_length must be equal to seq_output_length or seq_output_length must be 1.")
        
        self.dataframe = dataset_path if isinstance(dataset_path, pd.DataFrame) else pd.read_csv(dataset_path)

        if 'Timestamp' in self.dataframe.columns:
            self.dataframe.loc[:, 'Timestamp'] = pd.to_datetime(self.dataframe.loc[:, 'Timestamp'])
            self.dataframe = self.dataframe.sort_values(by=['MMSI', 'Timestamp']).reset_index(drop=True)
        
        self.seq_input_length = seq_input_length
        self.seq_output_length = seq_output_length
        self.valid_idxs = []
        

        # --- 1. INDEXING LOGIC ---
        # Precompute valid indices
        unique_mmsis = self.dataframe.loc[:, 'MMSI'].unique()
        for mmsi in unique_mmsis:
            mmsi_mask = self.dataframe.loc[:, 'MMSI'] == mmsi
            mmsi_df = self.dataframe[mmsi_mask]
            
            # Iterate through segments
            for segment_id, segment_data in mmsi_df.groupby('Segment'):
                if len(segment_data) < (self.seq_input_length + self.seq_output_length):
                   continue
                indices = segment_data.index
                num_sequences = len(segment_data) - (self.seq_input_length + self.seq_output_length) + 1
                for i in range(num_sequences):
                    self.valid_idxs.append(indices[i])

        # --- 2. NORMALIZATION LOGIC ---
        # If no stats are provided, compute from current dataset
        if stats is None:
            self.lat_min = self.dataframe.loc[:, 'Latitude'].min()
            self.lat_max = self.dataframe.loc[:, 'Latitude'].max()
            self.lon_min = self.dataframe.loc[:, 'Longitude'].min()
            self.lon_max = self.dataframe.loc[:, 'Longitude'].max()
            self.sog_max = self.dataframe.loc[:, 'SOG'].max() # Add speed max
        else:
            self.lat_min, self.lat_max, self.lon_min, self.lon_max, self.sog_max = stats

        # Save stats so we can retrieve them later
        self.stats = (self.lat_min, self.lat_max, self.lon_min, self.lon_max, self.sog_max)

        # Apply Normalization
        # epsilon 1e-6 to avoid division by zero
        self.dataframe.loc[:, 'Latitude'] = (self.dataframe.loc[:, 'Latitude'] - self.lat_min) / (self.lat_max - self.lat_min + 1e-6)
        self.dataframe.loc[:, 'Longitude'] = (self.dataframe.loc[:, 'Longitude'] - self.lon_min) / (self.lon_max - self.lon_min + 1e-6)
        
        # Normalize Speed (SOG) roughly 0-1
        self.dataframe.loc[:, 'SOG'] = self.dataframe.loc[:, 'SOG'] / (self.sog_max + 1e-6)           
                


    ### ================================================================
    ### --- PyTorch Dataset Required Methods ---
    ### ================================================================
    def __len__(self):
        """
        Returns:
            int: Number of valid input-output sequences.
        """
        return len(self.valid_idxs)

    ### ================================================================
    ### --- Input/Output Tensor Construction ---
    ### ================================================================
    def __getitem__(self, idx):
        """
        Constructs one input-output pair from the AIS data.

        Args:
            idx (int): Index of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of input tensor x and target tensor y.
        """
        real_idx = self.valid_idxs[idx]

        # --- 3. COG (Degrees to Radians) ---
        cog_deg = self.dataframe.iloc[real_idx: real_idx + self.seq_input_length]['COG'].to_numpy(dtype=float)
        cog_rad = np.deg2rad(cog_deg) # Convert to radians
        cog_tensor = torch.tensor(cog_rad, dtype=torch.float32)
        
        cog_sin = torch.sin(cog_tensor)
        cog_cos = torch.cos(cog_tensor)

        # x features
        x_data = self.dataframe.iloc[real_idx: real_idx + self.seq_input_length][['Latitude', 'Longitude', 'SOG']].to_numpy(dtype=float)
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        
        # Concatenate: Lat, Lon, SOG, Sin, Cos
        x = torch.cat((x_tensor, cog_sin.unsqueeze(-1), cog_cos.unsqueeze(-1)), dim=-1)

        # y labels
        y_data = self.dataframe.iloc[real_idx + self.seq_input_length: 
                                     real_idx + self.seq_input_length + self.seq_output_length][['Latitude', 'Longitude']].to_numpy(dtype=float)
        y = torch.tensor(y_data, dtype=torch.float32)

        return x, y
