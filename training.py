import torch
from torch import dropout
from torch.utils.data import DataLoader
from modules.dataset import AISDataset 
# from modules.dataset_deltas import AISDataset --- IGNORE ---
from modules.models import GRUModel
import matplotlib.pyplot as plt
from modules.losses import HaversineLoss
from tqdm import tqdm

if __name__ == "__main__":


    #GLOBAL HYPERPARAMETERS
    sequence_input_length = 5
    sequence_output_length = 5
    batch_size = 64 
    dropout_num = 0.2 #FOR THE DROPOUT LAYER IN THE MODEL
    lr = 0.00001 #LEARNING RATE FOR ADAM OPTIMIZER
    num_epochs = 2 #NUMBER OF EPOCHS TO TRAIN
    patience = 3 #EARLY STOPPING PATIENCE

    trainset = AISDataset('datasplits/train.csv', seq_input_length=sequence_input_length, seq_output_length=sequence_output_length)
    
    # 2. Extract stats from Train Set
    train_stats = trainset.stats
    
    # 3. Pass stats to Validation Set
    valset = AISDataset('datasplits/val.csv', seq_input_length=sequence_input_length, seq_output_length=sequence_output_length, stats=train_stats)
    # Create data loaders


    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=4)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    
    num_layers = [2] #[2, 4]
    embedding_sizes = [64] #[64, 128]
    hidden_size = [64] #[64, 128]
    training_losses = {}
    validation_losses = {}
    early_stopping_epochs = {}
    val_losses_i = {}
    train_losses_i = {}    
    # Test all combinations of hyperparameters
    for nl in num_layers:
        for es in embedding_sizes:
            for hs in hidden_size:
                print(f"Training with num_layers={nl}, embedding_size={es}, hidden_size={hs}, batch_seize={batch_size}, dropout={dropout_num}")
                model = GRUModel(input_size=5, embed_size=es, hidden_size=hs, output_size=2, num_layers=nl, dropout=dropout_num).to(device)

                # Define optimizer and loss function
                
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                # train_loss_fn = HaversineLoss(trainset.lat_min, trainset.lat_max,
                #                            trainset.lon_min, trainset.lon_max)
                # val_loss_fn = HaversineLoss(valset.lat_min, valset.lat_max,
                #                          valset.lon_min, valset.lon_max)
                train_loss_fn = torch.nn.L1Loss()
                val_loss_fn = torch.nn.L1Loss()
                train_loss_fn = torch.nn.MSELoss()
                val_loss_fn = torch.nn.MSELoss()

                # Training loop
                best_val_loss = float('inf')
                patience_counter = 0
                train_losses_list = []
                val_losses_list = []
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
                        l = train_loss_fn(output, target)
                        l.backward()
                        optimizer.step()
                        train_loss += l.item()
                        batch_bar.set_postfix(loss=l.item())

                    train_loss /= len(train_loader)
                    train_losses_list.append(train_loss)
                    print(f"Training Loss: {train_loss:.6f}")

                    # Validation step with tqdm
                    model.eval()
                    val_loss = 0
                    val_bar = tqdm(val_loader, desc="Validation", leave=False)
                    with torch.no_grad():
                        for data, target in val_bar:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            batch_loss = val_loss_fn(output, target).item()
                            val_loss += batch_loss
                            val_bar.set_postfix(loss=batch_loss)

                    val_loss /= len(val_loader)
                    val_losses_list.append(val_loss)
                    print(f"Validation Loss: {val_loss:.6f}")

                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        train_loss_model = train_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping triggered at epoch {epoch+1}.")
                            break

                    # Save the trained model
                    training_losses[f'gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout'] = train_loss_model
                    validation_losses[f'gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout'] = best_val_loss
                    early_stopping_epochs[f'gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout'] = epoch + 1 - patience_counter
                    # torch.save(model.state_dict(), f'results/models/gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout.pth')
                    torch.save(model.state_dict(), f'gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout.pth')
                train_losses_i[f'gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout'] = train_losses_list
                val_losses_i[f'gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout'] = val_losses_list

                # Graph training and validation loss
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, len(train_losses_i[f'gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout']) + 1), train_losses_i[f'gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout'], label='Training Loss')
                plt.plot(range(1, len(val_losses_i[f'gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout']) + 1), val_losses_i[f'gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Training and Validation Loss Over Epochs \n (Layers={nl}, Embed Size={es}, Hidden Size={hs}, Dropout={dropout_num}, Input Length={sequence_input_length}, Epochs={num_epochs})')
                plt.legend()
                plt.show()
                plt.savefig(f'results/epochs_graphs/loss_graph {nl} layer, embed_size = {es}, hiden size =  {hs}, dropout = {dropout_num}, input_length = {sequence_input_length}, epochs = {num_epochs}.png')
                print(f"Graph saved as 'loss_graph {nl} layer, embed_size = {es}, hiden size =  {hs}, dropout = {dropout_num}, input_length = {sequence_input_length}, epochs = {num_epochs}.png'")



                # ==========================================
                # NEW: FOLIUM MAP (PER CONFIGURATION)
                # ==========================================
                import folium

                print(f"Generating map for configuration: L{nl}_E{es}_H{hs}...")

                # 1. Get a batch from validation
                model.eval()
                data_batch, target_batch = next(iter(val_loader))
                data_batch, target_batch = data_batch.to(device), target_batch.to(device)
                
                # Plot up to 5 boats
                num_boats_to_plot = min(5, data_batch.shape[0])

                with torch.no_grad():
                    prediction_deltas = model(data_batch) # Output is Deltas

                # 2. Define Unnormalization (using the stats from trainset)
                # Stats structure: (lat_min, lat_max, lon_min, lon_max, sog_max)
                lat_min, lat_max, lon_min, lon_max, _ = trainset.stats

                def unnormalize_path(norm_path_np):
                    real_path = []
                    for point in norm_path_np:
                        # point[0] is Lat, point[1] is Lon
                        r_lat = point[0] * (lat_max - lat_min) + lat_min
                        r_lon = point[1] * (lon_max - lon_min) + lon_min
                        real_path.append([r_lat, r_lon])
                    return real_path

                # 3. Setup Map Center (Start of first boat)
                first_hist_end = data_batch[0, -1, 0:2].cpu().numpy()
                center_lat = first_hist_end[0] * (lat_max - lat_min) + lat_min
                center_lon = first_hist_end[1] * (lon_max - lon_min) + lon_min
                m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

                colors = ['blue', 'green', 'purple', 'orange', 'darkred']

                for i in range(num_boats_to_plot):
                    color = colors[i % len(colors)]
                    boat_id = f"Boat {i+1}"

                    # --- A. HISTORY (Standard) ---
                    # Input is [Lat, Lon, SOG, Sin, Cos]. We take 0:2
                    hist_norm = data_batch[i, :, 0:2].cpu().numpy()
                    hist_real = unnormalize_path(hist_norm)
                    
                    # The "Anchor" point (Last known location)
                    last_known_pt_norm = hist_norm[-1] 

                    # --- B. RECONSTRUCT PREDICTION (Delta -> Position) ---
                    # Prediction = Last_Known + Delta
                    pred_delta = prediction_deltas[i].cpu().numpy()
                    pred_path_norm = last_known_pt_norm + pred_delta
                    pred_real = unnormalize_path(pred_path_norm)

                    # --- C. RECONSTRUCT TARGET (Delta -> Position) ---
                    true_delta = target_batch[i].cpu().numpy()
                    true_path_norm = last_known_pt_norm + true_delta
                    true_real = unnormalize_path(true_path_norm)

                    # --- D. DRAW ---
                    # 1. History
                    folium.PolyLine(hist_real, color=color, weight=3, opacity=0.5, tooltip=f"{boat_id} History").add_to(m)
                    folium.CircleMarker(location=hist_real[-1], radius=4, color='black', fill=True).add_to(m)

                    # 2. Actual Future (Solid)
                    folium.PolyLine(true_real, color=color, weight=4, opacity=0.9, tooltip=f"{boat_id} Actual").add_to(m)

                    # 3. Predicted Future (Dashed & Connected)
                    # We connect the anchor to the first prediction point for visual continuity
                    connected_pred = [hist_real[-1]] + pred_real
                    folium.PolyLine(connected_pred, color=color, weight=3, opacity=1, dash_array='10', tooltip=f"{boat_id} Pred").add_to(m)

                # 4. Save Map with Unique Name
                map_filename = f"results/maps/gru_MAE_model_{nl}layers_{es}embSize_{hs}hiddSize_{batch_size}batch_{dropout_num}dropout.html"
                m.save(map_filename)
                print(f"Map saved as '{map_filename}'")



                        
    # Write in a json file the training and validation losses
    import json
    losses = {
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'early_stopping_epochs': early_stopping_epochs,
        'train_losses_i': train_losses_i,
        'val_losses_i': val_losses_i
    }

    print("Final Losses:", losses)
    
    with open('results/losses.json', 'w') as f:
        json.dump(losses, f)

    # Graph training and validation loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
    # plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss Over Epochs \n ')
    # plt.legend()
    # plt.show()



    