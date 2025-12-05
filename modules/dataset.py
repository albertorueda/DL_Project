from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np

class AISDataset(Dataset):
    def __init__(self, dataset_path: str, seq_input_length: int = 6, seq_output_length: int = 1, stats=None):
        # We only support seq_output_length of 1 (autoregressive) or equal to seq_input_length (same window size input-output) for now
        if (seq_input_length != seq_output_length and seq_output_length != 1):
            raise ValueError("Either seq_input_length must be equal to seq_output_length or seq_output_length must be 1.")
        
        self.dataframe = dataset_path if isinstance(dataset_path, pd.DataFrame) else pd.read_csv(dataset_path)

        if 'Timestamp' in self.dataframe.columns:
            self.dataframe['Timestamp'] = pd.to_datetime(self.dataframe['Timestamp'])
            self.dataframe = self.dataframe.sort_values(by=['MMSI', 'Timestamp']).reset_index(drop=True)
        
        self.seq_input_length = seq_input_length
        self.seq_output_length = seq_output_length
        self.valid_idxs = []
        

        # --- 1. INDEXING LOGIC ---
        # Precompute valid indices
        unique_mmsis = self.dataframe['MMSI'].unique()
        for mmsi in unique_mmsis:
            mmsi_mask = self.dataframe['MMSI'] == mmsi
            mmsi_df = self.dataframe[mmsi_mask]
            
            # Iterate through segments
            for segment_id, segment_data in mmsi_df.groupby('Segment'):
                if len(segment_data) < (self.seq_input_length + self.seq_output_length):
                   continue
                indices = segment_data.index
                num_sequences = len(segment_data) - (self.seq_input_length + self.seq_output_length) + 1
                for i in range(num_sequences):
                    self.valid_idxs.append(indices[i])

        # --- 2. NORMALIZATION LOGIC (FIXED) ---
        if stats is None:
            self.lat_min = self.dataframe['Latitude'].min()
            self.lat_max = self.dataframe['Latitude'].max()
            self.lon_min = self.dataframe['Longitude'].min()
            self.lon_max = self.dataframe['Longitude'].max()
            self.sog_max = self.dataframe['SOG'].max() # Add speed max
        else:
            self.lat_min, self.lat_max, self.lon_min, self.lon_max, self.sog_max = stats

        # Save stats so we can retrieve them later
        self.stats = (self.lat_min, self.lat_max, self.lon_min, self.lon_max, self.sog_max)

        # Apply Normalization
        # epsilon 1e-6 to avoid division by zero
        self.dataframe['Latitude'] = (self.dataframe['Latitude'] - self.lat_min) / (self.lat_max - self.lat_min + 1e-6)
        self.dataframe['Longitude'] = (self.dataframe['Longitude'] - self.lon_min) / (self.lon_max - self.lon_min + 1e-6)
        
        # Normalize Speed (SOG) roughly 0-1
        self.dataframe['SOG'] = self.dataframe['SOG'] / (self.sog_max + 1e-6)           
                


    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx):
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
