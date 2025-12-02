import math
from torch.utils.data import DataLoader
from modules.dataset import AISDataset

def loss_to_km(val_loss, lat_min, lat_max, lon_min, lon_max):
    # Rangos en grados
    R_lat = lat_max - lat_min
    R_lon = lon_max - lon_min
    
    # Latitud media para corregir longitud
    lat_mean = (lat_max + lat_min) / 2
    cos_lat = math.cos(math.radians(lat_mean))
    
    # Conversión a km por eje
    km_lat = val_loss * R_lat * 111
    km_lon = val_loss * R_lon * 111 * cos_lat
    
    # Media de ambos errores
    return (km_lat + km_lon) / 2

trainset = AISDataset('datasplits/train.csv', seq_input_length=5, seq_output_length=5)
    
    # 2. Extract stats from Train Set
train_stats = trainset.stats
    
    # 3. Pass stats to Validation Set
valset = AISDataset('datasplits/val.csv', seq_input_length=5, seq_output_length=5, stats=train_stats)
    
# Create data loaders
train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(valset, batch_size=64, shuffle=True, num_workers=4)
    
lat_min = min(trainset.lat_min, valset.lat_min)
lat_max = max(trainset.lat_max, valset.lat_max)
lon_min = min(trainset.lon_min, valset.lon_min)
lon_max = max(trainset.lon_max, valset.lon_max)

val_losses = [
    0.0042,  # num_layers=2, hidden=64
    0.0037,  # num_layers=2, hidden=128
    0.0034,  # num_layers=2, hidden=256
    0.0056,  # num_layers=4, hidden=64
    0.0047,  # num_layers=4, hidden=128
    0.0039   # num_layers=4, hidden=256
]

for i, loss in enumerate(val_losses, start=1):
    km_error = loss_to_km(
        val_loss=loss,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max
    )
    print(f"Exp {i}: Val Loss = {loss:.4f} → MAE ≈ {km_error:.2f} km")
