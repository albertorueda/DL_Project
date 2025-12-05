"""
finetuning_loop_batch_size.py

This script performs an ablation study on the effect of different batch sizes 
on the performance of a GRU model trained on AIS trajectory data.

It trains the model with fixed architecture and hyperparameters, using L1 loss,
early stopping, and logs the best validation/train loss for each batch size.

Results are stored in JSON format under 'results/' directory.
"""

# ================================================================
# --- IMPORTS ---
# ================================================================
import os
import json
import torch
from torch.utils.data import DataLoader
from modules.dataset import AISDataset 
from modules.models import GRUModel
from modules.losses import HaversineLoss  # Unused here, but kept if experimentation follows

# ================================================================
# --- MAIN EXECUTION ---
# ================================================================
if __name__ == "__main__":
    """
    Run training loop for different batch sizes and log train/val loss.
    Uses fixed GRU architecture, L1 loss, early stopping, and Adam optimizer.
    """

    # ================================================================
    # --- HYPERPARAMETERS AND SETUP ---
    # ================================================================
    batch_sizes = [16, 32, 64] 
    lr = 1e-5
    num_epochs = 1000
    patience = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    validation_loss_dict = {}
    train_loss_dict = {}
    
    # Load datasets
    trainset = AISDataset(os.path.join('datasplits', 'train.csv'), seq_input_length=5, seq_output_length=5)
    train_stats = trainset.stats
    valset = AISDataset(os.path.join('datasplits', 'val.csv'), seq_input_length=5, seq_output_length=5, stats=train_stats)
    
    loss_fn = torch.nn.L1Loss()
        
    # ================================================================
    # --- TRAINING LOOP FOR EACH BATCH SIZE ---
    # ================================================================
    for batchsize in batch_sizes:
        print(f"\n--- Training with batch size: {batchsize} ---")

        model = GRUModel(input_size=5, embed_size=64, hidden_size=256, output_size=2, num_layers=2, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=1)
        val_loader = DataLoader(valset, batch_size=batchsize, shuffle=True, num_workers=1)

        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses_list = []
        val_losses_list = []

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                l = loss_fn(output, target)
                l.backward()
                optimizer.step()
                train_loss += l.item()

            train_loss /= len(train_loader)
            train_losses_list.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += loss_fn(output, target).item()

            val_loss /= len(val_loader)
            val_losses_list.append(val_loss)

            if epoch % 10 == 0:
                print(f"    Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}: Best Val Loss: {best_val_loss:.4f}, Train Loss: {best_train_loss:.4f}")
                    break

        validation_loss_dict[f"batchsize_{batchsize}"] = best_val_loss
        train_loss_dict[f"batchsize_{batchsize}"] = best_train_loss

    # ================================================================
    # --- SAVE RESULTS ---
    # ================================================================
    os.makedirs("results", exist_ok=True)
    with open(os.path.join('results', 'validation_loss_batchsize.json'), 'w') as f:
        json.dump(validation_loss_dict, f, indent=4)
    with open(os.path.join('results', 'train_loss_batchsize.json'), 'w') as f:
        json.dump(train_loss_dict, f, indent=4)