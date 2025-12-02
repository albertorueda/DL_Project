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
    
    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load test dataset
    testset = AISDataset('datasplits/test.csv', seq_input_length=5, seq_output_length=5)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=1)
    lat_min = testset.lat_min
    lat_max = testset.lat_max
    lon_min = testset.lon_min
    lon_max = testset.lon_max
    
    model.eval()
    total_loss = 0
    
    # Open folder results/models 
    # Name of the model files: results/models/{model_type}_{loss}_{n_layers}layers_{h_size}hiddSize_.pth
    for filename in os.listdir('results/models'):
        model, loss, n_layers, hidden_size, _ = filename.split('_')
        if loss == "MAE":
            loss_fn = torch.nn.L1Loss() 
        else:
            loss_fn = HaversineLoss(lat_min, lat_max, lon_min, lon_max)
        
        if model == "GRU":
            model = GRUModel(input_size=5, embed_size=64, hidden_size=int(hidden_size.replace('hiddSize', '')), output_size=2, num_layers=int(n_layers.replace('layers', '')), dropout=0.2).to(device)
        elif model == "LSTM":
            model = LSTMModel(input_size=5, embed_size=64, hidden_size=int(hidden_size.replace('hiddSize', '')), output_size=2, num_layers=int(n_layers.replace('layers', '')), dropout=0.2).to(device)
            
        model.load_state_dict(torch.load(f'results/models/{filename}', map_location=device))
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
    
    average_loss = total_loss / len(test_loader)
    print(f"Average Test Loss: {average_loss:.4f}")
    
    ## Example prediction (print decoded coordinates and real locations)
    #it = iter(test_loader)
    ## first
    #sample_data, sample_target = next(it)
    #sample_data = sample_data.to(device)
    #sample_output = model(sample_data)
    #decoded_output = decode_predictions(sample_output, lat_min, lat_max, lon_min, lon_max)
    #decoded_target = decode_predictions(sample_target, lat_min, lat_max, lon_min, lon_max)
    #print("Sample decoded predictions (latitude, longitude):")
    #print(decoded_output[0].detach().cpu().numpy())
    #print("Sample real locations (latitude, longitude):")
    #print(decoded_target[0].detach().cpu().numpy())
    #
    ## next 2
    #sample_data, sample_target = next(it)
    #sample_data = sample_data.to(device)
    #sample_output = model(sample_data)
    #decoded_output = decode_predictions(sample_output, lat_min, lat_max, lon_min, lon_max)
    #decoded_target = decode_predictions(sample_target, lat_min, lat_max, lon_min, lon_max)
    #print("Next sample decoded predictions (latitude, longitude):")
    #print(decoded_output[1].detach().cpu().numpy())
    #print("Next sample real locations (latitude, longitude):")
    #print(decoded_target[1].detach().cpu().numpy())
    #
    #sample_data, sample_target = next(it)
    #sample_data = sample_data.to(device)
    #sample_output = model(sample_data)
    #decoded_output = decode_predictions(sample_output, lat_min, lat_max, lon_min, lon_max)
    #decoded_target = decode_predictions(sample_target, lat_min, lat_max, lon_min, lon_max)
    #print("Another sample decoded predictions (latitude, longitude):")
    #print(decoded_output[2].detach().cpu().numpy())
    #print("Another sample real locations (latitude, longitude):")
    #print(decoded_target[2].detach().cpu().numpy())