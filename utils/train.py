import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset import get_dataloaders
from models.model import RNN

def train_model(data_dir, epochs=10, batch_size=16, lr=0.001):
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNN(num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.squeeze(1).transpose(1, 2)  # (B, T, F)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Modello salvato come model.pth")
