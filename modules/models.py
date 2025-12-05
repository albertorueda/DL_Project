"""
models.py

Contains neural network model definitions for sequence-to-sequence prediction of vessel trajectories.
Includes GRUModel and LSTMModel classes with optional initial linear embedding.
"""

import torch.nn as nn

# for now : 
input_size = 4
output_size = 2 

### GRU for sequence-to-sequence 
class GRUModel(nn.Module):
    '''
    A GRU-based sequence-to-sequence model for predicting future positions based on input sequences.
    Args:
        input_size (int): The number of expected features in the input (default is 5).
        embed_size (int): The size of the embedding layer (default is 64).
        hidden_size (int): The number of features in the hidden state of the GRU (default is 256).
        output_size (int): The number of expected features in the output (default is 2).
        num_layers (int): Number of hidden layers in the GRU (default is 1).
        dropout (float): Dropout probability for the GRU layers (default is 0.1).
        first_linear (bool): Whether to include a linear embedding layer before the GRU (default is True).
    Returns:
        torch.Tensor: The output tensor containing predicted positions with shape (batch_size, seq_length, output_size).
    '''
    def __init__(self, input_size = 5, embed_size = 64, hidden_size = 256, output_size = 2, num_layers=1, dropout=0.1, first_linear=True):
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
    '''
    A LSTM-based sequence-to-sequence model for predicting future positions based on input sequences.
    Args:
        input_size (int): The number of expected features in the input (default is 5).
        embed_size (int): The size of the embedding layer (default is 64).
        hidden_size (int): The number of features in the hidden state of the LSTM (default is 256).
        output_size (int): The number of expected features in the output (default is 2).
        num_layers (int): Number of hidden layers in the LSTM (default is 1).
        dropout (float): Dropout probability for the LSTM layers (default is 0.0).
        first_linear (bool): Whether to include a linear embedding layer before the LSTM (default is True).
    Returns:
        torch.Tensor: The output tensor containing predicted positions with shape (batch_size, seq_length, output_size).

    '''
    def __init__(self, input_size = 5, embed_size = 64, hidden_size = 256, output_size = 2, num_layers=1, dropout=0.0, first_linear=True):
        super().__init__()
        self.first_linear = first_linear

        if self.first_linear:
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
        if self.first_linear:
            x = self.embedding(x)        # (batch, seq, 64)
        out, (hidden, cell) = self.lstm(x)    # (batch, seq, 64)
        out = self.fc_out(out)       # (batch, seq, 2)

        return out


