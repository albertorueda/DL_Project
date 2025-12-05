from modules.models import GRUModel, LSTMModel
import torch
from modules.dataset import AISDataset  
import pandas as pd
from modules.losses import HaversineLoss
import os

def decode_predictions(predictions, lat_min, lat_max, lon_min, lon_max):
    """Decode normalized predictions back to original latitude and longitude."""
    decoded = predictions.clone()
    decoded[..., 0] = decoded[..., 0] * (lat_max - lat_min) + lat_min  # Latitude
    decoded[..., 1] = decoded[..., 1] * (lon_max - lon_min) + lon_min  # Longitude
    return decoded

if __name__ == "__main__":
    
    # --------------------------------------------
    # LOAD TRAIN NORMALIZATION CONSTANTS (IMPORTANT!)
    # Ensures consistency between train / val / test
    # --------------------------------------------
    
    # Open datasplits folder
    data_files = os.listdir('datasplits/')
    
    # Concat all the csv that start with train
    train_files = ['datasplits/' + f for f in data_files if f.startswith('train') and f != 'train.csv']
    for i, file_path in enumerate(train_files):
        if i == 0:
            train_df = pd.read_csv(file_path)
        else:
            temp_df = pd.read_csv(file_path)
            train_df = pd.concat([train_df, temp_df], ignore_index=True)
    
    val_files = ['datasplits/' + f for f in data_files if f.startswith('val') and f != 'val.csv']
    for i, file_path in enumerate(val_files):
        if i == 0:
            val_df = pd.read_csv(file_path)
        else:
            temp_df = pd.read_csv(file_path)
            val_df = pd.concat([val_df, temp_df], ignore_index=True)
        
    lat_min = min(train_df['Latitude'].min(), val_df['Latitude'].min())
    lat_max = max(train_df['Latitude'].max(), val_df['Latitude'].max())
    lon_min = min(train_df['Longitude'].min(), val_df['Longitude'].min())
    lon_max = max(train_df['Longitude'].max(), val_df['Longitude'].max())
    sog_max = max(train_df['SOG'].max(), val_df['SOG'].max())

    # --------------------------------------------
    # Load model (ensure same architecture as training)
    # - Model 1: LSTM with Haversine Loss -- 2 layers, 256 hidden, 64 embed
    # - Model 2: GRU with MAE Loss -- 2 layers, 256 hidden, 64 embed
    # --------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = LSTMModel( #LSTMModel(
        input_size=5,
        embed_size=64,
        hidden_size=256,
        output_size=2,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load('results/models/more_days_hav.pth', map_location=device))

    # --------------------------------------------
    # Load test dataset (use train normalization stats)
    # --------------------------------------------
    # Concat all the csv that start with train
    test_files = ['datasplits/' + f for f in data_files if f.startswith('test') and f != 'test.csv']
    for i, file_path in enumerate(test_files):
        if i == 0:
            test_df = pd.read_csv(file_path)
        else:
            temp_df = pd.read_csv(file_path)
            test_df = pd.concat([test_df, temp_df], ignore_index=True)
            
    testset = AISDataset(
        test_df,
        seq_input_length=3,
        seq_output_length=3,
        stats=(lat_min, lat_max, lon_min, lon_max, sog_max)
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=1
    )

    # --------------------------------------------
    # Evaluate with Haversine Loss
    # --------------------------------------------
    model.eval()
    total_loss = 0
    loss_fn = HaversineLoss(lat_min, lat_max, lon_min, lon_max)

    print("Evaluating model...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sanity check
            assert output.shape == target.shape, \
                f"Shape mismatch: {output.shape} vs {target.shape}"

            loss = loss_fn(output, target)
            total_loss += loss.item()
    
    average_loss = total_loss / len(test_loader)
    print(f"Average Test Loss: {average_loss:.4f}")
    
    # --------------------------------------------
    # Example predictions decoding
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