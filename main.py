import torch
import torch.optim as optim
import torch.nn as nn
from utils.train_immagini import train_model
from utils.test_immagini import test_model
from utils.dataset_immagini import ImmaginiDataset
from models.model_immagini import CNNModel

if __name__ == "__main__":
    batch_size = 32
    img_size = 224
    num_epochs = 50
    learning_rate = 0.001
    patience = 10
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)

    train_dataset = ImmaginiDataset(root_dir='./dataset_diviso', subset='train', img_size=224)
    val_dataset   = ImmaginiDataset(root_dir='./dataset_diviso', subset='val', img_size=224)
    test_dataset  = ImmaginiDataset(root_dir='./dataset_diviso', subset='test', img_size=224)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    

    model = CNNModel(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=num_epochs, device=device, patience=patience)

    print("\nCaricamento del modello migliore...")
    model.load_state_dict(torch.load('best_model.pth'))

    print("\nEsecuzione del test sul dataset di test...")
    test_dataset.classes = [c for c in test_dataset.classes if not c.startswith('.')]
    test_model(test_loader, model, device=device, class_names=test_dataset.classes)
