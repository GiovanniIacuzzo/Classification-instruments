import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, num_classes=5, input_size=1, hidden_size=128, num_layers=2):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # bidirectional -> *2

    def forward(self, x):
        # x: (B, L)
        x = x.unsqueeze(-1)  # (B, L) -> (B, L, 1)
        out, _ = self.rnn(x)  # out: (B, L, 2H)
        out = out[:, -1, :]   # prendi l'ultimo timestep
        out = self.fc(out)    # (B, num_classes)
        return out
