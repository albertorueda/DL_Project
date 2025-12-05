"""
evaluate_more_data.py

This script evaluates a pre-trained LSTM model on AIS trajectory data using
Haversine loss. It loads normalization constants from training and validation
datasets to ensure consistent scaling, loads the test dataset, performs model
evaluation, and decodes predictions back to latitude and longitude coordinates.

Usage:
    python evaluate_more_data.py

Note:
    - The model and datasets paths are configurable via flags defined at the top.
    - This script only performs evaluation and prints results; it does not save
      or overwrite any models or outputs.
"""

from modules.models import LSTMModel
import torch
from modules.dataset import AISDataset  
import pandas as pd
from modules.losses import HaversineLoss
import os

# Configuration flags
MODEL_TYPE = 'LSTM'  # Options: 'LSTM', 'GRU' (currently only LSTM implemented)
MODEL_PATH = os.path.join('results', 'models', 'more_days_hav.pth')
TRAIN_DATA_DIR = os.path.join('datasplits', 'train')
VAL_DATA_DIR = os.path.join('datasplits', 'val')
TEST_DATA_DIR = os.path.join('datasplits', 'test')
SEQ_INPUT_LENGTH = 3
SEQ_OUTPUT_LENGTH = 3
BATCH_SIZE = 32
NUM_WORKERS = 1

def decode_predictions(predictions, lat_min, lat_max, lon_min, lon_max):
    """
    Decode normalized predictions back to original latitude and longitude coordinates.

    Args:
        predictions (torch.Tensor): Normalized predicted coordinates with shape (..., 2),
                                    where the last dimension represents (latitude, longitude).
        lat_min (float): Minimum latitude used for normalization.
        lat_max (float): Maximum latitude used for normalization.
        lon_min (float): Minimum longitude used for normalization.
        lon_max (float): Maximum longitude used for normalization.

    Returns:
        torch.Tensor: Decoded predictions with original latitude and longitude scale.
    """
    decoded = predictions.clone()
    decoded[..., 0] = decoded[..., 0] * (lat_max - lat_min) + lat_min  # Latitude
    decoded[..., 1] = decoded[..., 1] * (lon_max - lon_min) + lon_min  # Longitude
    return decoded

if __name__ == "__main__":
    
    # --------------------------------------------
    # Load train and validation datasets to obtain normalization constants
    # --------------------------------------------
    
    # Load and concatenate all training CSV files starting with 'train_'
    train_files = [os.path.join(TRAIN_DATA_DIR, f) for f in os.listdir(TRAIN_DATA_DIR) if f.startswith('train_') and f.endswith('.csv')]
    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True) if train_files else pd.DataFrame()

    # Load and concatenate all validation CSV files starting with 'val_'
    val_files = [os.path.join(VAL_DATA_DIR, f) for f in os.listdir(VAL_DATA_DIR) if f.startswith('val_') and f.endswith('.csv')]
    val_df = pd.concat([pd.read_csv(f) for f in val_files], ignore_index=True) if val_files else pd.DataFrame()
        
    # Compute normalization constants from train and val data combined
    lat_min = min(train_df['Latitude'].min(), val_df['Latitude'].min())
    lat_max = max(train_df['Latitude'].max(), val_df['Latitude'].max())
    lon_min = min(train_df['Longitude'].min(), val_df['Longitude'].min())
    lon_max = max(train_df['Longitude'].max(), val_df['Longitude'].max())
    sog_max = max(train_df['SOG'].max(), val_df['SOG'].max())

    # --------------------------------------------
    # Load the pre-trained model with the specified architecture
    # --------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if MODEL_TYPE == 'LSTM':
        model = LSTMModel(
            input_size=5,
            embed_size=64,
            hidden_size=256,
            output_size=2,
            num_layers=2,
            dropout=0.1,
        ).to(device)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # --------------------------------------------
    # Load and prepare test dataset using train/val normalization stats
    # --------------------------------------------
    test_files = [os.path.join(TEST_DATA_DIR, f) for f in os.listdir(TEST_DATA_DIR) if f.startswith('test_') and f.endswith('.csv')]
    test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True) if test_files else pd.DataFrame()
            
    testset = AISDataset(
        test_df,
        seq_input_length=SEQ_INPUT_LENGTH,
        seq_output_length=SEQ_OUTPUT_LENGTH,
        stats=(lat_min, lat_max, lon_min, lon_max, sog_max)
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # --------------------------------------------
    # Evaluate the model on the test data using Haversine loss
    # --------------------------------------------
    model.eval()
    total_loss = 0
    loss_fn = HaversineLoss(lat_min, lat_max, lon_min, lon_max)

    print("Evaluating model on test dataset...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Ensure output shape matches target shape
            assert output.shape == target.shape, \
                f"Shape mismatch: {output.shape} vs {target.shape}"

            loss = loss_fn(output, target)
            total_loss += loss.item()
    
    average_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('nan')
    print(f"Average Test Loss: {average_loss:.4f}")
    
    # --------------------------------------------
    # Decode and display example predictions alongside real trajectories
    # --------------------------------------------
    sample_data, sample_target = next(iter(test_loader))
    sample_data = sample_data.to(device)
    sample_output = model(sample_data)
    decoded_output = decode_predictions(sample_output, lat_min, lat_max, lon_min, lon_max)
    decoded_target = decode_predictions(sample_target, lat_min, lat_max, lon_min, lon_max)
    
    for i in range(len(decoded_output)):
        print("---------- Vessel Number " + str(i + 1) + " ----------")
        print(f"Real trajectory:")
        print(decoded_target[i].detach().cpu().numpy())
        print(f"Predicted trajectory:")
        print(decoded_output[i].detach().cpu().numpy())