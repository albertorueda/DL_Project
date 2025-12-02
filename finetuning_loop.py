import torch
from torch import dropout
from torch.utils.data import DataLoader
from modules.dataset import AISDataset 
from modules.models import GRUModel, LSTMModel
from modules.losses import HaversineLoss

if __name__ == "__main__":

    #GLOBAL HYPERPARAMETERS
    window_sizes = [3, 5, 7]
    batch_sizes = [16, 32, 64] 
    dropout_nums = [0.1, 0.2, 0.3, 0.4] #FOR THE DROPOUT LAYER IN THE MODEL
    lr = 0.00001 #LEARNING RATE FOR ADAM OPTIMIZER
    num_epochs = 100 #NUMBER OF EPOCHS TO TRAIN
    patience = 5 #EARLY STOPPING PATIENCE

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    validation_loss_dict = {}
    train_loss_dict = {}
    loss_fn = HaversineLoss(lat_min, lat_max, lon_min, lon_max)
    
    for window_size in window_sizes:
        # TODO: change to proper model and loss (the best one)
        model = LSTMModel(input_size=5, embed_size=64, hidden_size=, output_size=2, num_layers=, dropout=0.2).to(device)
        model.load_state_dict(torch.load(f'results/models/modelname.pth', map_location=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        print(f"\n--- Training {window_size} window size ---")
        trainset = AISDataset('datasplits/train.csv', seq_input_length=window_size, seq_output_length=window_size)
    
        # 2. Extract stats from Train Set
        train_stats = trainset.stats

        # 3. Pass stats to Validation Set
        valset = AISDataset('datasplits/val.csv', seq_input_length=window_size, seq_output_length=window_size, stats=train_stats)

        # Create data loaders
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=1)

        lat_min = min(trainset.lat_min, valset.lat_min)
        lat_max = max(trainset.lat_max, valset.lat_max)
        lon_min = min(trainset.lon_min, valset.lon_min)
        lon_max = max(trainset.lon_max, valset.lon_max)

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
        
    
    # Select best validation loss after this study
    best_window_size = int(min(validation_loss_dict, key=validation_loss_dict.get)[-1])
    print(f"Best window size: {best_window_size} with validation loss: {validation_loss_dict[f'window_size{best_window_size}']:.4f}")
    
    trainset = AISDataset('datasplits/train.csv', seq_input_length=best_window_size, seq_output_length=best_window_size)
    # 2. Extract stats from Train Set
    train_stats = trainset.stats
    # 3. Pass stats to Validation Set
    valset = AISDataset('datasplits/val.csv', seq_input_length=best_window_size, seq_output_length=best_window_size, stats=train_stats)
    lat_min = min(trainset.lat_min, valset.lat_min)
    lat_max = max(trainset.lat_max, valset.lat_max)
    lon_min = min(trainset.lon_min, valset.lon_min)
    lon_max = max(trainset.lon_max, valset.lon_max)
        
    # Now batchsize
    for batchsize in batch_sizes:
        # TODO: change to proper model and loss (the best one)
        model = LSTMModel(input_size=5, embed_size=64, hidden_size=, output_size=2, num_layers=, dropout=0.2).to(device)
        model.load_state_dict(torch.load(f'results/models/modelname.pth', map_location=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = HaversineLoss(lat_min, lat_max, lon_min, lon_max)
        
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
        
    # Select best bacthsize after study
    best_batchsize = min([validation_loss_dict["batchsize16"], validation_loss_dict["batchsize32"], validation_loss_dict["batchsize64"]])
    best_batchsize_key = int([key for key, value in validation_loss_dict.items() if value == best_batchsize and "batchsize" in key][0][-2:])
    print(f"Best batch size: {best_batchsize} with validation loss: {validation_loss_dict[f'batchsize{best_batchsize_key}']:.4f}")
        
    train_loader = DataLoader(trainset, batch_size=best_batchsize, shuffle=True, num_workers=1)
    val_loader = DataLoader(valset, batch_size=best_batchsize, shuffle=True, num_workers=1)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Lastly, dropout
    for dp in dropout_nums:
        print(f"\n--- Training {dp} dropout ---")
        model = LSTMModel(input_size=5, embed_size=64, hidden_size=, output_size=2, num_layers=, dropout=dp).to(device)
        model.load_state_dict(torch.load(f'results/models/modelname.pth', map_location=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
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
        
    # save validation and training loss dictionaries in a json file
    with open('results/validation_loss.json', 'w') as f:
        json.dump(validation_loss_dict, f)
    with open('results/train_loss.json', 'w') as f:
        json.dump(train_loss_dict, f)
    