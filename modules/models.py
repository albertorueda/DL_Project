import torch.nn as nn
from torch.nn import GRU, Module, Linear, ReLU, Sequential, Dropout

## for now : 
input_size = 4
output_size = 2 

### GRU for sequence-to-sequence 
class GRUModel(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers=1, dropout=0.0, first_linear=True):
        super().__init__()
        self.first_linear = first_linear

        if first_linear:
            # Map input from size 5 (input_size) to size 64 (embed_size)
            self.embedding = nn.Linear(input_size, embed_size)
            # GRU works in 64-dim space 
            self.gru = nn.GRU(
                embed_size,
                hidden_size,
                num_layers,
                bias=True,
                batch_first=True,
                dropout = dropout ,
                bidirectional=False
            )
        else:
            # GRU works in 4-dim space 
            self.gru = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                bias=True,
                batch_first=True,
                dropout = dropout ,
                bidirectional=False
            )

        # Map GRU outputs from size 64 (embed_size) to size 2 (input/output_size) again 
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.first_linear:
            x = self.embedding(x)        # (batch, seq, 64)
        out, hidden = self.gru(x)    # (batch, seq, 64)
        out = self.fc_out(out)       # (batch, seq, 2)

        return out


### LSTM for sequence-to-sequence

class LSTMModel(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers=1, dropout=0.0, first_linear=True):
        super().__init__()

        if first_linear:
            # Map input from size 5 (input_size) to size 64 (embed_size)
            self.embedding = nn.Linear(input_size, embed_size)

            # LSTM works in 64-dim space 
            self.lstm = nn.LSTM(
                embed_size,
                hidden_size,
                num_layers,
                bias=True,
                batch_first=True,
                dropout = dropout ,
                bidirectional=False
            )
        
        else:
            # LSTM works in 4-dim space 
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                bias=True,
                batch_first=True,
                dropout = dropout ,
                bidirectional=False
            )

        # Map LSTM outputs from size 64 (embed_size) to size 2 (input/output_size) again 
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)        # (batch, seq, 64)
        out, (hidden, cell) = self.lstm(x)    # (batch, seq, 64)
        out = self.fc_out(out)       # (batch, seq, 2)

        return out


## we will not be using this for now :
class GRUSeq2VecModel(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()

        # Map input from size 4 (input_size) to size 64 (embed_size)
        self.embedding = nn.Linear(input_size, embed_size)

        # GRU works in 64-dim space 
        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            num_layers,
            bias=True,
            batch_first=True,
            dropout = dropout ,
            bidirectional=False
        )

        # Map GRU final hidden state from size 64 (hidden_size) to size 2 (output_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)        # (batch, seq, 64)
        out, hidden = self.gru(x)    # out: (batch, seq, 64), hidden: (num_layers, batch, 64)
        out = self.fc_out(hidden[-1]) # Take the last layer's hidden state and map to output size

        return out

