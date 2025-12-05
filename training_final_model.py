import torch
from torch import dropout
from torch.utils.data import DataLoader
from modules.dataset import AISDataset 
from modules.models import GRUModel, LSTMModel
import matplotlib.pyplot as plt
from modules.losses import HaversineLoss
from tqdm import tqdm
import json

if __name__ == "__main__":

    #GLOBAL HYPERPARAMETERS
    sequence_input_length = 3
    sequence_output_length = 3
    batch_size = 32 
    dropout_num = 0.1 #FOR THE DROPOUT LAYER IN THE MODEL
    lr = 0.00001 #LEARNING RATE FOR ADAM OPTIMIZER
    num_epochs = 1000 #NUMBER OF EPOCHS TO TRAIN
    patience = 5 #EARLY STOPPING PATIENCE
    
    type_of_loss = 'MAE'

    trainset = AISDataset('datasplits/train.csv', seq_input_length=sequence_input_length, seq_output_length=sequence_output_length)
    
    # 2. Extract stats from Train Set
    train_stats = trainset.stats
    
    # 3. Pass stats to Validation Set
    valset = AISDataset('datasplits/val.csv', seq_input_length=sequence_input_length, seq_output_length=sequence_output_length, stats=train_stats)
    
    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    lat_min = min(trainset.lat_min, valset.lat_min)
    lat_max = max(trainset.lat_max, valset.lat_max)
    lon_min = min(trainset.lon_min, valset.lon_min)
    lon_max = max(trainset.lon_max, valset.lon_max)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    training_losses = {}
    validation_losses = {}
    early_stopping_epochs = {}
    val_losses_i = {}
    train_losses_i = {}    

    model = GRUModel(input_size=5, embed_size=64, hidden_size=256, output_size=2, num_layers=2, dropout=0.1).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if type_of_loss == 'HAVERSINE': 
        loss_fn = HaversineLoss(lat_min, lat_max, lon_min, lon_max)
    elif type_of_loss == 'MAE':
        loss_fn = torch.nn.L1Loss()

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
            l = loss_fn(output, target)
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
                batch_loss = loss_fn(output, target).item()
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
    torch.save(model.state_dict(), 'results/models/mae_final_model.pth')
    
    # Save training and validation losses
    training_losses['final_model'] = train_losses_list
    validation_losses['final_model'] = val_losses_list
    early_stopping_epochs['final_model'] = epoch + 1
    
    with open('results/results_final_model_mae.json', 'w') as f:
        json.dump({
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'early_stopping_epochs': early_stopping_epochs
        }, f)
    
    print("Training complete. Model and results saved.")