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
    
    test_df = pd.read_csv('datasplits/test.csv')
    lat_min = test_df['Latitude'].min()
    lat_max = test_df['Latitude'].max()
    lon_min = test_df['Longitude'].min()
    lon_max = test_df['Longitude'].max()
    
    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GRUModel(input_size=5, embed_size=32, hidden_size=32, output_size=2, num_layers=2, dropout=0.2, first_linear=False).to(device)
    model.load_state_dict(torch.load('results/models/gru_model_2_32_32_False.pth', map_location=device))
    
    # Load test dataset
    testset = AISDataset('datasplits/test.csv', seq_input_length=3, seq_output_length=3)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)
    
    model.eval()
    total_loss = 0
    loss_fn = HaversineLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
    
    average_loss = total_loss / len(test_loader)
    print(f"Average Test Loss: {average_loss:.4f}")
    
    # Example prediction (print decoded coordinates and real locations)
    sample_data, sample_target = next(iter(test_loader))
    sample_data = sample_data.to(device)
    sample_output = model(sample_data)
    decoded_output = decode_predictions(sample_output, lat_min, lat_max, lon_min, lon_max)
    decoded_target = decode_predictions(sample_target, lat_min, lat_max, lon_min, lon_max)
    print("Sample decoded predictions (latitude, longitude):")
    print(decoded_output[0].detach().cpu().numpy())
    print("Sample real locations (latitude, longitude):")
    print(decoded_target[0].detach().cpu().numpy())