from torch.utils.data import Dataset
import pandas as pd
import torch

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
            # Get all rows for this MMSI for each Segment except the last seq_input_length + seq_output_length - 1 rows
            mmsi_data = self.dataframe[self.dataframe['MMSI'] == mmsi].groupby('Segment').apply(lambda x: x, include_groups=False).reset_index(drop=True)
            if len(mmsi_data) < (self.seq_input_length + self.seq_output_length):
                continue
            # Add the indexes of valid starting points for sequences
            for start_idx in range(len(mmsi_data) - (self.seq_input_length + self.seq_output_length) + 1):
                real_idx = mmsi_data.index[start_idx]
                self.valid_idxs.append(real_idx)
            
                
        # Normalization parameters
        self.lat_min = self.dataframe['Latitude'].min()
        self.lat_max = self.dataframe['Latitude'].max()
        self.lon_min = self.dataframe['Longitude'].min()
        self.lon_max = self.dataframe['Longitude'].max()
        
        self.dataframe['Latitude'] = (self.dataframe['Latitude'] - self.lat_min) / (self.lat_max - self.lat_min)
        self.dataframe['Longitude'] = (self.dataframe['Longitude'] - self.lon_min) / (self.lon_max - self.lon_min)

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx):
        real_idx = self.valid_idxs[idx]

        # Get sin and cos of COG for better representation
        cog = self.dataframe.iloc[real_idx: real_idx + self.seq_input_length]['COG'].to_numpy(dtype=float)
        cog = torch.tensor(cog, dtype=torch.float32) 
        cog_sin = torch.sin(cog)
        cog_cos = torch.cos(cog)

        # x features
        x = self.dataframe.iloc[real_idx: real_idx + self.seq_input_length][['Latitude', 'Longitude', 'SOG']].to_numpy(dtype=float)
        x = torch.tensor(x, dtype=torch.float32)      
        x = torch.cat((x, cog_sin.unsqueeze(-1), cog_cos.unsqueeze(-1)), dim=-1)

        # y labels
        y = self.dataframe.iloc[real_idx + self.seq_input_length: 
                                real_idx + self.seq_input_length + self.seq_output_length][['Latitude', 'Longitude']].to_numpy(dtype=float)
        y = torch.tensor(y, dtype=torch.float32)      

        return x, y
