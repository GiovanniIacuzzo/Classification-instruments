import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.evaluate_audio import evaluate

def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-3, device=None, save_path="best_model.pth"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []
    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("Modello salvato (nuovo best)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Nessun miglioramento da {epochs_no_improve} epoche.")

        if epochs_no_improve >= patience:
            print(f"Early stopping attivato dopo {epoch + 1} epoche.")
            break


        scheduler.step()

        print(f"\nFine training. Miglior acc. validazione: {best_val_acc:.4f}")
    return all_train_loss, all_val_loss, all_train_acc, all_val_acc
