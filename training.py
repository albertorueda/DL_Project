import torch
from torch.utils.data import DataLoader
from modules.dataset import AISDataset
from modules.models import GRUModel

if __name__ == "__main__":
    # Prepare datasets and data loaders
    trainset = AISDataset('data/train.csv', seq_input_length=3, seq_output_length=3)
    valset = AISDataset('data/val.csv', seq_input_length=3, seq_output_length=3)

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = GRUModel(input_size=4, hidden_size=64, output_size=2*3, num_layers=2, dropout=0.2).to(device)

    # Define optimizer and loss function
    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.MSELoss()

    # Training loop (simplified)
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            l = loss(output, target)
            print(f"        Batch {batch_idx+1}/{len(train_loader)} - Loss: {l.item()}")
            l.backward()
            optimizer.step()
            train_loss += l.item()

        train_loss /= len(train_loader)
        print(f"    Training Loss: {train_loss}")

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += loss(output, target).item()

        val_loss /= len(val_loader)
        print(f"    Validation Loss: {val_loss}")

    # Save the trained model
    torch.save(model.state_dict(), 'gru_model.pth')