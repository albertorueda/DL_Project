"""
training_big_loop.py

This script performs a grid search over LSTM and GRU models with varying configurations
(number of layers, hidden size, loss function) for vessel trajectory prediction.
It uses AISDataset for input preprocessing and supports both MAE and Haversine losses.
Results (validation and train loss) are saved to JSON files in the results/ folder.
"""

### ================================================================
### --- IMPORTS ---
### ================================================================
import os
import torch
from torch.utils.data import DataLoader
from modules.dataset import AISDataset 
from modules.models import GRUModel, LSTMModel
from modules.losses import HaversineLoss
import json

if __name__ == "__main__":

    ### ================================================================
    ### --- DATA LOADING ---
    ### ================================================================
    # GLOBAL HYPERPARAMETERS
    sequence_input_length = 5
    sequence_output_length = 5
    batch_size = 64
    dropout_num = 0.2  # FOR THE DROPOUT LAYER IN THE MODEL
    lr = 0.00001  # LEARNING RATE FOR ADAM OPTIMIZER
    num_epochs = 1000  # NUMBER OF EPOCHS TO TRAIN
    patience = 5  # EARLY STOPPING PATIENCE

    loss_types = ['MAE', 'HAVERSINE']
    models_to_test = ['LSTM', 'GRU']

    trainset = AISDataset(os.path.join('datasplits', 'train_aisdk-2025-02-27.csv'),
                          seq_input_length=sequence_input_length,
                          seq_output_length=sequence_output_length)

    # 2. Extract stats from Train Set
    train_stats = trainset.stats

    # 3. Pass stats to Validation Set
    valset = AISDataset(os.path.join('datasplits', 'val_aisdk-2025-02-27.csv'),
                        seq_input_length=sequence_input_length,
                        seq_output_length=sequence_output_length,
                        stats=train_stats)

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=1)

    lat_min = min(trainset.lat_min, valset.lat_min)
    lat_max = max(trainset.lat_max, valset.lat_max)
    lon_min = min(trainset.lon_min, valset.lon_min)
    lon_max = max(trainset.lon_max, valset.lon_max)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    num_layers = [2, 4]
    embedding_sizes = 64
    hidden_size = [64, 128, 256]
    validation_loss_dict = {}
    train_loss_dict = {}

    ### ================================================================
    ### --- GRID SEARCH TRAINING LOOP ---
    ### ================================================================
    # Test all combinations of hyperparameters
    for model_type in models_to_test:
        print(f"\n--- Training {model_type} Models ---")

        for loss_type in loss_types:
            print(f"    Using {loss_type} loss function.")
            if loss_type == 'MAE':
                loss_fn = torch.nn.L1Loss()
            elif loss_type == 'HAVERSINE':
                loss_fn = HaversineLoss(lat_min, lat_max, lon_min, lon_max)

            for n_layers in num_layers:
                for h_size in hidden_size:
                    # input_size=5 corresponds to 3 numeric features + 2 trigonometric COG features
                    if model_type == 'GRU':
                        model = GRUModel(input_size=5, embed_size=embedding_sizes, hidden_size=h_size,
                                         output_size=2, num_layers=n_layers, dropout=dropout_num,
                                         first_linear=False).to(device)
                    elif model_type == 'LSTM':
                        model = LSTMModel(input_size=5, embed_size=embedding_sizes, hidden_size=h_size,
                                          output_size=2, num_layers=n_layers, dropout=dropout_num,
                                          first_linear=False).to(device)
                    print(f"        With num_layers={n_layers}, embedding_size={embedding_sizes}, hidden_size={h_size}, "
                          f"batch_size={batch_size}, dropout={dropout_num}")

                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    # Training loop
                    best_val_loss = float('inf')
                    patience_counter = 0
                    train_losses_list = []
                    val_losses_list = []
                    best_train_loss = None
                    for epoch in range(num_epochs):
                        model.train()
                        train_loss = 0

                        # Training step
                        for batch_idx, (data, target) in enumerate(train_loader):
                            data, target = data.to(device), target.to(device)
                            optimizer.zero_grad()
                            output = model(data)
                            l = loss_fn(output, target)
                            l.backward()
                            optimizer.step()
                            train_loss += l.item()

                        train_loss /= len(train_loader)
                        train_losses_list.append(train_loss)
                        if (epoch + 1) % 10 == 0:
                            print(f"        Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

                        # Validation step
                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for data, target in val_loader:
                                data, target = data.to(device), target.to(device)
                                output = model(data)
                                batch_loss = loss_fn(output, target).item()
                                val_loss += batch_loss

                        val_loss /= len(val_loader)
                        val_losses_list.append(val_loss)

                        # Early stopping check
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_train_loss = train_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"        Early stopping triggered at epoch {epoch + 1}: "
                                      f"Best Val Loss: {best_val_loss:.4f}, Train Loss: {best_train_loss:.4f}")
                                break

                    # Save the trained model losses
                    validation_loss_dict[(model_type, loss_type, n_layers, h_size)] = best_val_loss
                    train_loss_dict[(model_type, loss_type, n_layers, h_size)] = best_train_loss

    ### ================================================================
    ### --- SAVE RESULTS TO JSON ---
    ### ================================================================
    with open('results/validation_loss_divided.json', 'w') as f:
        json.dump(validation_loss_dict, f)
    with open('results/train_loss_divided.json', 'w') as f:
        json.dump(train_loss_dict, f)

    print("\n--- Training Complete ---")

### ================================================================
### --- END OF SCRIPT ---
### ================================================================