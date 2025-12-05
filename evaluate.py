from modules.models import GRUModel, LSTMModel
import torch
from modules.dataset import AISDataset  
import pandas as pd
from modules.losses import HaversineLoss

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
    train_df = pd.read_csv('datasplits/train.csv')
    val_df = pd.read_csv('datasplits/val.csv')
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

    model_gru = GRUModel( #GRUModel(
        input_size=5,
        embed_size=64,
        hidden_size=256,
        output_size=2,
        num_layers=2,
        dropout=0.1,
    ).to(device)
    
    model_lstm = LSTMModel( #LSTMModel(
        input_size=5,
        embed_size=64,
        hidden_size=256,
        output_size=2,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    model_gru.load_state_dict(torch.load('results/models/mae_final_model.pth', map_location=device))
    model_lstm.load_state_dict(torch.load('results/models/hav_final_model.pth', map_location=device))

    # --------------------------------------------
    # Load test dataset (use train normalization stats)
    # --------------------------------------------
    testset = AISDataset(
        'datasplits/test.csv',
        seq_input_length=3,
        seq_output_length=3,
        stats=(lat_min, lat_max, lon_min, lon_max, sog_max)
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=1
    )

    # --------------------------------------------
    # Evaluate with Haversine Loss (now consistent)
    # --------------------------------------------
    #model.eval()
    #total_loss = 0
    #loss_fn = HaversineLoss(lat_min, lat_max, lon_min, lon_max)
    #loss_fn = torch.nn.L1Loss()
#
    #print("Evaluating model...")
    #with torch.no_grad():
    #    for data, target in test_loader:
    #        data, target = data.to(device), target.to(device)
    #        output = model(data)
#
    #        # Sanity check
    #        assert output.shape == target.shape, \
    #            f"Shape mismatch: {output.shape} vs {target.shape}"
#
    #        loss = loss_fn(output, target)
    #        total_loss += loss.item()
    #
    #average_loss = total_loss / len(test_loader)
    #print(f"Average Test Loss: {average_loss:.4f}")
    
    # --------------------------------------------
    # Three example predictions decoding
    # --------------------------------------------
    
    it = iter(test_loader)
    
    # First
    sample_data, sample_target = next(it)
    sample_data = sample_data.to(device)
    sample_output_gru = model_gru(sample_data)
    sample_output_lstm = model_lstm(sample_data)
    decoded_output_gru = decode_predictions(sample_output_gru, lat_min, lat_max, lon_min, lon_max)
    decoded_output_lstm = decode_predictions(sample_output_lstm, lat_min, lat_max, lon_min, lon_max)
    decoded_target = decode_predictions(sample_target, lat_min, lat_max, lon_min, lon_max)
    print("Sample decoded predictions (latitude, longitude) - GRU:")
    print(decoded_output_gru[0].detach().cpu().numpy())
    print("Sample decoded predictions (latitude, longitude) - LSTM:")
    print(decoded_output_lstm[0].detach().cpu().numpy())
    print("Sample real locations (latitude, longitude):")
    print(decoded_target[0].detach().cpu().numpy())
    
    # Second
    sample_data, sample_target = next(it)
    sample_data = sample_data.to(device)
    sample_output_gru = model_gru(sample_data)
    sample_output_lstm = model_lstm(sample_data)
    decoded_output_gru = decode_predictions(sample_output_gru, lat_min, lat_max, lon_min, lon_max)
    decoded_output_lstm = decode_predictions(sample_output_lstm, lat_min, lat_max, lon_min, lon_max)
    decoded_target = decode_predictions(sample_target, lat_min, lat_max, lon_min, lon_max)
    print("Next sample decoded predictions (latitude, longitude) - GRU:")
    print(decoded_output_gru[1].detach().cpu().numpy())
    print("Next sample decoded predictions (latitude, longitude) - LSTM:")
    print(decoded_output_lstm[1].detach().cpu().numpy())
    print("Next sample real locations (latitude, longitude):")
    print(decoded_target[1].detach().cpu().numpy())
    
    # Third
    sample_data, sample_target = next(it)
    sample_data = sample_data.to(device)
    sample_output_gru = model_gru(sample_data)
    sample_output_lstm = model_lstm(sample_data)
    decoded_output_gru = decode_predictions(sample_output_gru, lat_min, lat_max, lon_min, lon_max)
    decoded_output_lstm = decode_predictions(sample_output_lstm, lat_min, lat_max, lon_min, lon_max)
    decoded_target = decode_predictions(sample_target, lat_min, lat_max, lon_min, lon_max)
    print("Another sample decoded predictions (latitude, longitude) - GRU:")
    print(decoded_output_gru[2].detach().cpu().numpy())
    print("Another sample decoded predictions (latitude, longitude) - LSTM:")
    print(decoded_output_lstm[2].detach().cpu().numpy())
    print("Another sample real locations (latitude, longitude):")
    print(decoded_target[2].detach().cpu().numpy())
