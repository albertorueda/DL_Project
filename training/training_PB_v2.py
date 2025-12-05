import torch
from torch import dropout
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.dataset import AISDataset
from modules.models import GRUModel
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Initialization of Train Set
    seq_input_length = 5
    seq_output_length = 5

    trainset = AISDataset('train.csv', seq_input_length=seq_input_length, seq_output_length=seq_output_length)
    
    # 2. Extract stats from Train Set
    train_stats = trainset.stats
    
    # 3. Pass stats to Validation Set
    valset = AISDataset('val.csv', seq_input_length=seq_input_length, seq_output_length=seq_output_length, stats=train_stats)

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    n_l = 2
    embed_size = 64
    hidden_size = 64
    dropout_num = 0.2
    model = GRUModel(input_size=5, embed_size=embed_size, hidden_size=hidden_size, output_size=2, num_layers=n_l, dropout=dropout_num).to(device)

    # Define optimizer and loss function
    lr = 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.L1Loss()
    

    # Training loop
    num_epochs = 2
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        model.train()
        train_loss = 0

        # Training step with tqdm over batches
        batch_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (data, target) in enumerate(batch_bar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            l = loss(output, target)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            batch_bar.set_postfix(loss=l.item())

        train_loss /= len(train_loader)
        training_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.6f}")

        # Validation step with tqdm
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for data, target in val_bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                batch_loss = loss(output, target).item()
                val_loss += batch_loss
                val_bar.set_postfix(loss=batch_loss)

        val_loss /= len(val_loader)
        validation_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Save the trained model
    torch.save(model.state_dict(), 'gru_model_30nov.pth')

    # Graph training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Over Epochs (Layers={n_l}, Embed Size={embed_size}, Hidden Size={hidden_size}, Dropout={dropout_num}, Input Length={seq_input_length}, Epochs={num_epochs})')
    plt.legend()
    plt.show()
    plt.savefig(f'loss_graph {n_l} layer, embed_size = {embed_size}, hiden size =  {hidden_size}, dropout = {dropout_num}, input_length = {seq_input_length}, epochs = {num_epochs}.png')
    print(f"Graph saved as 'loss_graph {n_l} layer, embed_size = {embed_size}, hiden size =  {hidden_size}, dropout = {dropout_num}, input_length = {seq_input_length}, epochs = {num_epochs}.png'")





# ==========================================
# NEW: FOLIUM MAP FOR 5 BOATS
# ==========================================
import folium
import matplotlib.colors as mcolors

print("Generating trajectory map for 5 boats...")

# 1. Get a batch and predict
model.eval()
# Ensure batch size is at least 5
data_batch, target_batch = next(iter(val_loader))
data_batch, target_batch = data_batch.to(device), target_batch.to(device)

num_boats_to_plot = min(4, data_batch.shape[0]) # Plot 5, or fewer if batch is small

with torch.no_grad():
    prediction_batch = model(data_batch)

# 2. Define Unnormalization Functions & Stats
lat_min, lat_max, lon_min, lon_max, _ = valset.stats

def unnormalize_path(norm_path_np):
    """Helper to unnormalize a list of [lat, lon] points"""
    real_path = []
    for point in norm_path_np:
        r_lat = point[0] * (lat_max - lat_min) + lat_min
        r_lon = point[1] * (lon_max - lon_min) + lon_min
        real_path.append([r_lat, r_lon])
    return real_path

# 3. Setup Map & Colors
# Center map on the end of the first boat's history
first_hist_end = data_batch[0, -1, 0:2].cpu().numpy()
center_lat = first_hist_end[0] * (lat_max - lat_min) + lat_min
center_lon = first_hist_end[1] * (lon_max - lon_min) + lon_min
m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

# Distinct colors for different boats
colors = ['blue', 'green', 'purple', 'orange', 'darkred']

# 4. Loop through boats and plot
for i in range(0, 8, 4):
    if i % 4 != 0:
        continue
    color = colors[i % len(colors)] # Cycle colors if more than 5
    boat_id = f"Boat {i+1}"

# ... Inside the loop for boats ...
    
    # --- A. Extract & Unnormalize History (Input) ---
    hist_norm = data_batch[i, :, 0:2].cpu().numpy()
    hist_real = unnormalize_path(hist_norm)

    # --- B. Extract & Unnormalize True Future (Target) ---
    # NOTE: target_batch is now DELTAS, so we need to reconstruct the position
    # But for the graph, let's just cheat and calculate the real future from the dataset logic or 
    # simply use the visual connection. 
    # Actually, let's just stick to the simplest visual check:
    
    # Let's get the "Last Known Point" in normalized coords
    last_known_pt_norm = hist_norm[-1] 

    # --- C. Handle Prediction (RECONSTRUCTION) ---
    pred_delta_norm = prediction_batch[i].cpu().numpy()
    
    # RECONSTRUCT: Prediction = Last_Known + Delta
    # Note: pred_delta_norm is shape (SeqLen, 2)
    # We add the delta to the last known point to find where the ship went
    pred_pos_norm = last_known_pt_norm + pred_delta_norm
    
    # Now unnormalize the reconstructed path
    pred_real = unnormalize_path(pred_pos_norm)
    
    # Unnormalize the True Future (for comparison)
    # Since we modified __getitem__, target_batch is deltas. 
    # We should reconstruct the true path similarly:
    true_delta_norm = target_batch[i].cpu().numpy()
    true_pos_norm = last_known_pt_norm + true_delta_norm
    true_real = unnormalize_path(true_pos_norm)

    # --- D. Draw on Map ---
    # History (Solid Line)
    folium.PolyLine(hist_real, color=color, weight=3, opacity=0.5).add_to(m)
    
    # Connection Dot (The "Anchor")
    folium.CircleMarker(location=hist_real[-1], radius=4, color='black', fill=True).add_to(m)

    # True Future (Solid Line)
    folium.PolyLine(true_real, color=color, weight=4, opacity=0.9).add_to(m)

    # Predicted Future (Dashed Line) - NOW CONNECTED!
    folium.PolyLine(pred_real, color=color, weight=3, opacity=1, dash_array='10').add_to(m)


# 5. Save Map
map_filename = "multi_boat_trajectory_2.html"
m.save(map_filename)
print(f"Map saved as '{map_filename}'.")