import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.dataset import AISDataset
from modules.models import GRUModel
import matplotlib.pyplot as plt
from modules.losses import HaversineLoss

if __name__ == "__main__":    
    # Prepare datasets and data loaders
    trainset = AISDataset('datasplits/train.csv', seq_input_length=3, seq_output_length=3)
    valset = AISDataset('datasplits/val.csv', seq_input_length=3, seq_output_length=3)

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = GRUModel(input_size=5, embed_size=64, hidden_size=64, output_size=2, num_layers=2, dropout=0.2).to(device)

    # Define optimizer and loss function
    lr = 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = HaversineLoss()

    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    training_losses = []
    validation_losses = []

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
            l = loss(output, target)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            batch_bar.set_postfix(loss=l.item())

        train_loss /= len(train_loader)
        training_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation step with tqdm
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for data, target in val_bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                batch_loss = loss(output, target).item()
                val_loss += batch_loss
                val_bar.set_postfix(loss=batch_loss)

        val_loss /= len(val_loader)
        validation_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Save the trained model
    torch.save(model.state_dict(), 'gru_model.pth')

    # Graph training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()