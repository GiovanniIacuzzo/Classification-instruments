import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, num_classes=5):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        output, _ = self.rnn(x)
        out = output[:, -1, :]
        return self.classifier(out)