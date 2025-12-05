"""
Training script to evaluate the effect of different batch sizes on model performance.
Trains a GRU model on AIS trajectory data using L1 loss and early stopping.
Results are saved in JSON files under 'results/'.
"""
import os
import json
import torch
from torch.utils.data import DataLoader
from modules.dataset import AISDataset 
from modules.models import GRUModel, LSTMModel
from modules.losses import HaversineLoss

if __name__ == "__main__":
    """
    Run training loop for different batch sizes and log train/val loss.
    Uses fixed GRU architecture, L1 loss, early stopping, and Adam optimizer.
    Saves best losses for each batch size configuration.
    """

    # GLOBAL HYPERPARAMETERS
    batch_sizes = [16, 32, 64] 
    lr = 0.00001 # LEARNING RATE FOR ADAM OPTIMIZER
    num_epochs = 1000 # NUMBER OF EPOCHS TO TRAIN
    patience = 5 # EARLY STOPPING PATIENCE

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    validation_loss_dict = {}
    train_loss_dict = {}
    
    trainset = AISDataset(os.path.join('datasplits', 'train.csv'), seq_input_length=5, seq_output_length=5)
    # 2. Extract stats from Train Set
    train_stats = trainset.stats
    # 3. Pass stats to Validation Set
    valset = AISDataset(os.path.join('datasplits', 'val.csv'), seq_input_length=5, seq_output_length=5, stats=train_stats)
    
    loss_fn = torch.nn.L1Loss()
        
    # Now batchsize
    for batchsize in batch_sizes:
        # Using GRU model with fixed hyperparameters for batch size ablation study
        model = GRUModel(input_size=5, embed_size=64, hidden_size=256, output_size=2, num_layers=2, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        print(f"\n--- Training {batchsize} batchsize ---")

        # Create data loaders
        train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=1)
        val_loader = DataLoader(valset, batch_size=batchsize, shuffle=True, num_workers=1)

        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
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
        validation_loss_dict[f"batchsize{batchsize}"] = best_val_loss
        train_loss_dict[f"batchsize{batchsize}"] = train_loss_model

    # save validation and training loss dictionaries in a json file
    with open('results/validation_loss_batchsize.json', 'w') as f:
        json.dump(validation_loss_dict, f)
    with open('results/train_loss_batchsize.json', 'w') as f:
        json.dump(train_loss_dict, f)