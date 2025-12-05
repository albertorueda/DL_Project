"""
Fine-tuning loop to test different window sizes (input/output lengths) for sequence-to-sequence prediction models.
Trains a GRU model with fixed hyperparameters across different window sizes and tracks training/validation losses.
"""

### ================================================================
### --- IMPORTS ---
### ================================================================
import os
import json
import torch
from torch.utils.data import DataLoader
from modules.dataset import AISDataset 
from modules.models import GRUModel, LSTMModel  # Import available model architectures
from modules.losses import HaversineLoss

# ================================================================
# --- MAIN EXECUTION ---
# ================================================================

if __name__ == "__main__":

    # Loop over different window sizes and train a model for each configuration

    # ================================================================
    # --- HYPERPARAMETERS AND SETUP ---
    # ================================================================
    window_sizes = [3, 5, 7]
    lr = 0.00001 # LEARNING RATE FOR ADAM OPTIMIZER
    num_epochs = 1000 # NUMBER OF EPOCHS TO TRAIN
    patience = 5 # EARLY STOPPING PATIENCE

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    validation_loss_dict = {}
    train_loss_dict = {}
    
    for window_size in window_sizes:
        model = GRUModel(input_size=5, embed_size=64, hidden_size=256, output_size=2, num_layers=2, dropout=0.2, first_linear=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        print(f"\n--- Training window size {window_size} ---")

        train_path = os.path.join("datasplits/train", "train_aisdk-2025-02-27.csv")
        val_path = os.path.join("datasplits/val", "val_aisdk-2025-02-27.csv")
        trainset = AISDataset(train_path, seq_input_length=window_size, seq_output_length=window_size)
    
        # 2. Extract stats from Train Set
        train_stats = trainset.stats

        # 3. Pass stats to Validation Set
        valset = AISDataset(val_path, seq_input_length=window_size, seq_output_length=window_size, stats=train_stats)

        # Create data loaders
        train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
        val_loader = DataLoader(valset, batch_size=64, shuffle=True, num_workers=1)

        loss_fn = torch.nn.L1Loss()

        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
        ### ================================================================
        ### --- TRAINING LOOP ---
        ### ================================================================

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses_list = []
        val_losses_list = []
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            # Training step with tqdm over batches
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

            # Validation step with tqdm
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
            
            if epoch % 10 == 0:
                print(f"        Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                train_loss_model = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"        Early stopping triggered at epoch {epoch+1}: Best Val Loss: {best_val_loss:.4f}, Train Loss: {train_loss_model:.4f}")
                    break

        # Save the trained model
        validation_loss_dict[f"window_size{window_size}"] = best_val_loss
        train_loss_dict[f"window_size{window_size}"] = train_loss_model
        
    ### ================================================================
    ### --- SAVE LOSSES ---
    ### ================================================================

    # Save best validation and training losses for each window size configuration to results folder
    with open(os.path.join('results', 'validation_loss_window_size.json'), 'w') as f:
        json.dump(validation_loss_dict, f)
    with open(os.path.join('results', 'train_loss_window_size.json'), 'w') as f:
        json.dump(train_loss_dict, f)