from torch.nn import GRU, Module, Linear, ReLU, Sequential, Dropout

class GRUModel(Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.gru = GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = Sequential(
            Linear(hidden_size, hidden_size),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1]) # [batch_size, 6]
        out = out.view(out.size(0), 3, 2) # [batch_size, 3, 2]
        return out
