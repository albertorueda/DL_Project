from modules.models import GRUModel, LSTMModel
import torch
from modules.dataset import AISDataset  
import pandas as pd
from modules.losses import HaversineLoss
from modules.metrics import ADE, decode_predictions, FDE, RMSE

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

    model_gru = GRUModel(
        input_size=5,
        embed_size=64,
        hidden_size=256,
        output_size=2,
        num_layers=2,
        dropout=0.1,
    ).to(device)
    
    model_lstm = LSTMModel(
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
    model_lstm.eval()
    model_gru.eval()
    total_ade_lstm = 0
    total_fde_lstm = 0
    total_rmse_lstm = 0
    total_ade_gru = 0
    total_fde_gru = 0
    total_rmse_gru = 0

    print("Evaluating model...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_lstm = model_lstm(data)
            output_gru = model_gru(data)
            
            
            output_lstm = decode_predictions(output_lstm, lat_min, lat_max, lon_min, lon_max)
            output_gru = decode_predictions(output_gru, lat_min, lat_max, lon_min, lon_max)
            target = decode_predictions(target, lat_min, lat_max, lon_min, lon_max)

            # Sanity check
            assert output_lstm.shape == target.shape, \
                f"Shape mismatch: {output_lstm.shape} vs {target.shape}"
                
            total_ade_lstm += ADE(output_lstm, target)
            total_ade_gru += ADE(output_gru, target)
            total_fde_lstm += FDE(output_lstm, target)
            total_fde_gru += FDE(output_gru, target)
            total_rmse_lstm += RMSE(output_lstm, target)
            total_rmse_gru += RMSE(output_gru, target)
    
    average_ade_lstm = total_ade_lstm / len(test_loader)
    average_fde_lstm = total_fde_lstm / len(test_loader)
    average_rmse_lstm = total_rmse_lstm / len(test_loader)

    print(f"Average ADE LSTM: {average_ade_lstm:.4f}")
    print(f"Average FDE LSTM: {average_fde_lstm:.4f}")
    print(f"Average RMSE LSTM: {average_rmse_lstm:.4f}")
    
    average_ade_gru = total_ade_gru / len(test_loader)
    average_fde_gru = total_fde_gru / len(test_loader)
    average_rmse_gru = total_rmse_gru / len(test_loader)
    
    print(f"Average ADE GRU: {average_ade_gru:.4f}")
    print(f"Average FDE GRU: {average_fde_gru:.4f}")
    print(f"Average RMSE GRU: {average_rmse_gru:.4f}")
    