import torch
from torch.utils.data import DataLoader
from modules.dataset import AISDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
import folium

# FLAGS: Choose model type and loss function
MODEL_TYPE = "LSTM"  # Options: "LSTM", "GRU"
LOSS_TYPE = "HAVERSINE"  # Options: "HAVERSINE", "MAE", "HYBRID"

sequence_input_length = 5
sequence_output_length = 5
batch_size = 64
dropout_num = 0.2
lr = 0.00001
num_epochs = 100
patience = 5

num_layers_list = [2, 4, 8]
embedding_sizes = [64, 128]
hidden_sizes = [64, 128]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for nl in num_layers_list:
    for es in embedding_sizes:
        for hs in hidden_sizes:
            print(f"Starting training for configuration: {nl} layers, {es} embedding size, {hs} hidden size")

            # Load training dataset to get lat/lon min/max
            train_dataset = AISDataset("datasplits/train.csv", sequence_input_length, sequence_output_length)
            lat_min, lat_max = train_dataset.lat_min, train_dataset.lat_max
            lon_min, lon_max = train_dataset.lon_min, train_dataset.lon_max

            val_dataset = AISDataset("datasplits/val.csv", sequence_input_length, sequence_output_length,
                                     stats=train_dataset.stats)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Define model based on flag
            if MODEL_TYPE == "LSTM":
                from modules.models import LSTMModel
                model = LSTMModel(
                    input_size=5,
                    embed_size=es,
                    hidden_size=hs,
                    output_size=2,
                    num_layers=nl,
                    dropout=dropout_num,
                    first_linear=True
                ).to(device)
            elif MODEL_TYPE == "GRU":
                from modules.models import GRUModel
                model = GRUModel(
                    input_size=5,
                    embed_size=es,
                    hidden_size=hs,
                    output_size=2,
                    num_layers=nl,
                    dropout=dropout_num,
                    first_linear=True
                ).to(device)
            else:
                raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

            # Define loss function based on flag
            if LOSS_TYPE == "HAVERSINE":
                from modules.losses import HaversineLoss
                criterion = HaversineLoss(lat_min, lat_max, lon_min, lon_max).to(device)
            elif LOSS_TYPE == "MAE":
                criterion = torch.nn.L1Loss()
            elif LOSS_TYPE == "HYBRID":
                from modules.losses import HybridTrajectoryLoss
                criterion = HybridTrajectoryLoss().to(device)
            else:
                raise ValueError(f"Unsupported loss type: {LOSS_TYPE}")

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_val_loss = float('inf')
            epochs_no_improve = 0
            train_losses = []
            val_losses = []

            for epoch in range(1, num_epochs + 1):
                model.train()
                train_loss = 0
                for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    if LOSS_TYPE == "HYBRID":
                        loss = criterion(outputs, targets)[0]
                    else:
                        loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                train_loss /= len(train_loader.dataset)
                train_losses.append(train_loss)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        if LOSS_TYPE == "HYBRID":
                            loss = criterion(outputs, targets)[0]
                        else:
                            loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)

                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state = model.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            filename = f"{MODEL_TYPE.lower()}_{LOSS_TYPE}_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout.pth"
            model_dir = f"results/{MODEL_TYPE}_{LOSS_TYPE}/models"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(best_model_state, os.path.join(model_dir, filename))
            print(f"Saved best model to {filename}")

            losses_path = f"results/{MODEL_TYPE}_{LOSS_TYPE}/losses_{LOSS_TYPE}.json"
            if os.path.exists(losses_path):
                with open(losses_path, "r") as f:
                    all_losses = json.load(f)
            else:
                all_losses = {}

            config_key = f"{MODEL_TYPE}_{LOSS_TYPE}_{nl}layers_{es}embSize_{hs}hiddSize"
            all_losses[config_key] = {"train": train_losses, "val": val_losses}

            with open(losses_path, "w") as f:
                json.dump(all_losses, f)
            print(f"Saved losses to {losses_path}")

            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss\n{config_key}')
            plt.legend()
            graph_path = f"results/{MODEL_TYPE}_{LOSS_TYPE}/epochs_graphs/{config_key}_{batch_size}batch_{dropout_num}dropout.png"
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            plt.savefig(graph_path)
            plt.close()
            print(f"Saved epoch graph to {graph_path}")

            # Generate Folium map
            print("Generating Folium map...")
            m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=6)
            num_boats_to_plot = 4
            for idx in range(num_boats_to_plot):
                sample = val_dataset[idx]
                inputs, targets = sample
                inputs = inputs.unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    preds = model(inputs).cpu().numpy()[0]

                # Denormalize lat/lon
                def denormalize_lat(x): return x * (lat_max - lat_min) + lat_min
                def denormalize_lon(x): return x * (lon_max - lon_min) + lon_min

                true_points = [(denormalize_lat(lat.item()), denormalize_lon(lon.item())) for lat, lon in targets]
                pred_points = [(denormalize_lat(lat), denormalize_lon(lon)) for lat, lon in preds]

                folium.PolyLine(true_points, color="green", weight=2.5, opacity=1).add_to(m)
                folium.PolyLine(pred_points, color="red", weight=2.5, opacity=1).add_to(m)

            map_path = f"results/{MODEL_TYPE}_{LOSS_TYPE}/maps/{config_key}_map.html"
            os.makedirs(os.path.dirname(map_path), exist_ok=True)
            m.save(map_path)
            print(f"Saved Folium map to {map_path}")