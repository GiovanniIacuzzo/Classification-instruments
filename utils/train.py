import torch
import torch.nn as nn
import torch.optim as optim
from utils.get_data import get_dataloaders
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
        for i, (x, y) in enumerate(train_loader):
            print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}")
            
            # Trasferisci i dati sul dispositivo
            x, y = x.to(device), y.to(device)

            # Verifica la forma di x e y prima del passaggio alla GRU
            print(f"x.shape: {x.shape}, y.shape: {y.shape}")

            # Assicurati che x abbia la forma corretta: (B, L, F)
            # Se x è di forma (B, L), aggiungiamo una dimensione per le feature
            if x.dim() == 2:
                x = x.unsqueeze(-1)  # (B, L) -> (B, L, 1)

            # Controlla di nuovo la forma di x
            print(f"After unsqueeze, x.shape: {x.shape}")

            # Forward pass
            optimizer.zero_grad()
            outputs = model(x)

            # Debugging: controlla la forma dell'output
            print(f"outputs.shape: {outputs.shape}")
            
            # Calcola la loss
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Modello salvato come model.pth")
