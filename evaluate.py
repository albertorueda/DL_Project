from modules.models import GRUModel
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
    lat_min = train_df['Latitude'].min()
    lat_max = train_df['Latitude'].max()
    lon_min = train_df['Longitude'].min()
    lon_max = train_df['Longitude'].max()

    # --------------------------------------------
    # Load model (ensure same architecture as training)
    # --------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = GRUModel(
        input_size=5,
        embed_size=32,
        hidden_size=32,
        output_size=2,
        num_layers=2,
        dropout=0.2,
        first_linear=True   # ensure matches training!
    ).to(device)

    model.load_state_dict(torch.load('results/models/gru_model_2_32_32_False.pth', map_location=device))

    # --------------------------------------------
    # Load test dataset (use train normalization stats)
    # --------------------------------------------
    testset = AISDataset(
        'datasplits/test.csv',
        seq_input_length=3,
        seq_output_length=3,
        stats=(lat_min, lat_max, lon_min, lon_max, train_df['SOG'].max())
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=4
    )

    # --------------------------------------------
    # Evaluate with Haversine Loss (now consistent)
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
    print(f"\nAverage Test Loss (km): {average_loss:.4f}\n")
    
    # --------------------------------------------
    # Example decoded prediction
    # --------------------------------------------
    print("Example decoded prediction:")
    sample_data, sample_target = next(iter(test_loader))
    sample_data = sample_data.to(device)
    sample_output = model(sample_data)

    decoded_output = decode_predictions(sample_output, lat_min, lat_max, lon_min, lon_max)
    decoded_target = decode_predictions(sample_target, lat_min, lat_max, lon_min, lon_max)

    print("Predicted (lat, lon):")
    print(decoded_output[0].detach().cpu().numpy())

    print("\nReal (lat, lon):")
    print(decoded_target[0].detach().cpu().numpy())