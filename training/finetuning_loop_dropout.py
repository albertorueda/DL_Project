### ================================================================
### --- IMPORTS ---
### ================================================================
"""
Finetuning experiment: GRU model with varying dropout rates.
This script trains a GRU-based sequence model on AIS data using different dropout values.
It applies early stopping and saves the best training and validation losses for each configuration.

Output:
- Trained models (not saved here)
- validation_loss_dropout.json
- train_loss_dropout.json
"""
import torch
from torch.utils.data import DataLoader
from modules.dataset import AISDataset 
from modules.models import GRUModel
from torch.nn import L1Loss
import os
import json

# ================================================================
# --- MAIN EXECUTION ---
# ================================================================

if __name__ == "__main__":
    """
    Loop over dropout values [0.1, 0.2, 0.3, 0.4] and train a GRU model for each.
    Records best validation and training loss for each config using early stopping.
    """

    # ================================================================
    # --- HYPERPARAMETERS AND SETUP ---
    # ================================================================
    dropout_nums = [0.1, 0.2, 0.3, 0.4] # FOR THE DROPOUT LAYER IN THE MODEL
    lr = 0.00001 # LEARNING RATE FOR ADAM OPTIMIZER
    num_epochs = 1000 # NUMBER OF EPOCHS TO TRAIN
    patience = 5 # EARLY STOPPING PATIENCE

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    validation_loss_dict = {}
    train_loss_dict = {}
    
    trainset = AISDataset(os.path.join("datasplits", "train_aisdk-2025-02-27.csv"), seq_input_length=5, seq_output_length=5)
    # 2. Extract stats from Train Set
    train_stats = trainset.stats
    # 3. Pass stats to Validation Set
    valset = AISDataset(os.path.join("datasplits", "val_aisdk-2025-02-27.csv"), seq_input_length=5, seq_output_length=5, stats=train_stats)
    
    loss_fn = L1Loss()

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
    val_loader = DataLoader(valset, batch_size=64, shuffle=True, num_workers=1)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Lastly, dropout
    for dp in dropout_nums:
        print(f"\n--- Training {dp} dropout ---")
        model = GRUModel(input_size=5, embed_size=64, hidden_size=256, output_size=2, num_layers=2, dropout=dp).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        ### ================================================================
        ### --- TRAINING LOOP ---
        ### ================================================================
        
        # Training loop
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
        validation_loss_dict[f"dropout{dp}"] = best_val_loss
        train_loss_dict[f"dropout{dp}"] = train_loss_model
        
    ### ================================================================
    ### --- SAVE LOSSES ---
    ### ================================================================
    # save validation and training loss dictionaries in a json file
    os.makedirs(os.path.join("results", "finetuning"), exist_ok=True)
    with open(os.path.join("results", "finetuning", "validation_loss_dropout.json"), 'w') as f:
        json.dump(validation_loss_dict, f)
    with open(os.path.join("results", "finetuning", "train_loss_dropout.json"), 'w') as f:
        json.dump(train_loss_dict, f)