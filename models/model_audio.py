import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, num_classes=5, dropout=0.3, bidirectional=False):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.bn = nn.BatchNorm1d(hidden_size * self.num_directions)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out
