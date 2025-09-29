import torch
import torch.optim as optim
import torch.nn as nn
from utils.train_immagini import train_model
from utils.test_immagini import test_model
from utils.dataset_immagini import ImmaginiDataset
from models.model_immagini import CNNModel

if __name__ == "__main__":
    batch_size = 64
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


"""
(base) giovanni02@MacBook-Air-del-Professore Classification-instruments % /usr/local/bin/python3 /Users/giovanni02/Desktop/Progetti/Classifi
cation-instruments/main.py

Epoch [1/50]
Train Loss: 15.8326, Train Acc: 58.35%                                                                                                      
Val Loss: 10.9361, Val Acc: 54.91%

Epoch [2/50]
Train Loss: 3.6461, Train Acc: 77.35%                                                                                                       
Val Loss: 3.8957, Val Acc: 73.41%

Epoch [3/50]
Train Loss: 1.5649, Train Acc: 85.28%                                                                                                       
Val Loss: 4.1540, Val Acc: 76.30%

Epoch [4/50]
Train Loss: 1.0558, Train Acc: 86.33%                                                                                                       
Val Loss: 1.5769, Val Acc: 76.88%

Epoch [5/50]
Train Loss: 0.3686, Train Acc: 87.58%                                                                                                       
Val Loss: 1.6762, Val Acc: 80.35%

Epoch [6/50]
Train Loss: 0.3186, Train Acc: 88.62%                                                                                                       
Val Loss: 1.7355, Val Acc: 82.08%

Epoch [7/50]
Train Loss: 0.4412, Train Acc: 86.33%                                                                                                       
Val Loss: 1.4512, Val Acc: 79.19%

Epoch [8/50]
Train Loss: 0.3029, Train Acc: 88.83%                                                                                                       
Val Loss: 1.6125, Val Acc: 78.61%

Epoch [9/50]
Train Loss: 0.2798, Train Acc: 88.20%                                                                                                       
Val Loss: 1.9081, Val Acc: 78.61%

Epoch [10/50]
Train Loss: 0.2652, Train Acc: 90.92%                                                                                                       
Val Loss: 1.8789, Val Acc: 82.08%

Epoch [11/50]
Train Loss: 0.3205, Train Acc: 89.14%                                                                                                       
Val Loss: 1.6639, Val Acc: 83.24%

Epoch [12/50]
Train Loss: 0.2560, Train Acc: 89.98%                                                                                                       
Val Loss: 1.7430, Val Acc: 83.24%

Epoch [13/50]
Train Loss: 0.3312, Train Acc: 87.79%                                                                                                       
Val Loss: 1.5559, Val Acc: 81.50%

Epoch [14/50]
Train Loss: 0.2730, Train Acc: 90.19%                                                                                                       
Val Loss: 1.5545, Val Acc: 83.24%

Epoch [15/50]
Train Loss: 0.2179, Train Acc: 91.23%                                                                                                       
Val Loss: 1.4229, Val Acc: 84.97%

Epoch [16/50]
Train Loss: 0.2298, Train Acc: 91.34%                                                                                                       
Val Loss: 1.2727, Val Acc: 83.24%

Epoch [17/50]
Train Loss: 0.2056, Train Acc: 92.90%                                                                                                       
Val Loss: 1.3538, Val Acc: 84.97%

Epoch [18/50]
Train Loss: 0.1862, Train Acc: 94.15%                                                                                                       
Val Loss: 1.7075, Val Acc: 83.82%

Epoch [19/50]
Train Loss: 0.2193, Train Acc: 92.48%                                                                                                       
Val Loss: 1.3306, Val Acc: 84.39%

Epoch [20/50]
Train Loss: 0.1530, Train Acc: 93.95%                                                                                                       
Val Loss: 1.7196, Val Acc: 80.92%

Epoch [21/50]
Train Loss: 0.2277, Train Acc: 91.13%                                                                                                       
Val Loss: 1.2769, Val Acc: 83.82%

Epoch [22/50]
Train Loss: 0.1936, Train Acc: 92.69%                                                                                                       
Val Loss: 1.4976, Val Acc: 84.39%

Epoch [23/50]
Train Loss: 0.2781, Train Acc: 90.40%                                                                                                       
Val Loss: 1.3629, Val Acc: 83.24%

Epoch [24/50]
Train Loss: 0.1965, Train Acc: 92.48%                                                                                                       
Val Loss: 1.5408, Val Acc: 84.39%

Epoch [25/50]
Train Loss: 0.1584, Train Acc: 95.20%                                                                                                       
Val Loss: 1.0970, Val Acc: 85.55%

Epoch [26/50]
Train Loss: 0.2252, Train Acc: 91.86%                                                                                                       
Val Loss: 1.5446, Val Acc: 82.66%

Epoch [27/50]
Train Loss: 0.1932, Train Acc: 91.75%                                                                                                       
Val Loss: 1.8867, Val Acc: 86.13%

Epoch [28/50]
Train Loss: 0.1868, Train Acc: 93.74%                                                                                                       
Val Loss: 1.6581, Val Acc: 85.55%

Epoch [29/50]
Train Loss: 0.1539, Train Acc: 94.47%                                                                                                       
Val Loss: 1.2282, Val Acc: 85.55%

Epoch [30/50]
Train Loss: 0.1397, Train Acc: 94.78%                                                                                                       
Val Loss: 1.5233, Val Acc: 83.82%

Epoch [31/50]
Train Loss: 0.1009, Train Acc: 95.51%                                                                                                       
Val Loss: 2.2278, Val Acc: 86.71%

Epoch [32/50]
Train Loss: 0.1156, Train Acc: 95.72%                                                                                                       
Val Loss: 1.7122, Val Acc: 88.44%

Epoch [33/50]
Train Loss: 0.0984, Train Acc: 96.35%                                                                                                       
Val Loss: 1.7169, Val Acc: 87.28%

Epoch [34/50]
Train Loss: 0.1154, Train Acc: 95.20%                                                                                                       
Val Loss: 1.2143, Val Acc: 87.86%

Epoch [35/50]
Train Loss: 0.0999, Train Acc: 96.56%                                                                                                       
Val Loss: 1.2161, Val Acc: 89.02%

Epoch [36/50]
Train Loss: 0.1028, Train Acc: 95.72%                                                                                                       
Val Loss: 0.8458, Val Acc: 87.86%

Epoch [37/50]
Train Loss: 0.0904, Train Acc: 96.66%                                                                                                       
Val Loss: 1.1310, Val Acc: 88.44%

Epoch [38/50]
Train Loss: 0.0926, Train Acc: 96.87%                                                                                                       
Val Loss: 1.4538, Val Acc: 88.44%

Epoch [39/50]
Train Loss: 0.0935, Train Acc: 96.35%                                                                                                       
Val Loss: 1.3201, Val Acc: 88.44%

Epoch [40/50]
Train Loss: 0.1119, Train Acc: 96.03%                                                                                                       
Val Loss: 0.8750, Val Acc: 87.86%

Epoch [41/50]
Train Loss: 0.1273, Train Acc: 95.62%                                                                                                       
Val Loss: 0.8118, Val Acc: 89.02%

Epoch [42/50]
Train Loss: 0.0665, Train Acc: 97.08%                                                                                                       
Val Loss: 1.2738, Val Acc: 88.44%

Epoch [43/50]
Train Loss: 0.0682, Train Acc: 97.18%                                                                                                       
Val Loss: 0.9687, Val Acc: 87.86%

Epoch [44/50]
Train Loss: 0.1076, Train Acc: 96.03%                                                                                                       
Val Loss: 1.3153, Val Acc: 87.28%

Epoch [45/50]
Train Loss: 0.0948, Train Acc: 96.56%                                                                                                       
Val Loss: 1.6787, Val Acc: 86.71%
Early stopping triggered

Best Validation Accuracy: 89.02%

Final Evaluation Report:

              precision    recall  f1-score   support

           0     0.6727    1.0000    0.8043        37
           1     1.0000    0.9730    0.9863        37
           2     1.0000    0.9545    0.9767        22
           3     0.9024    0.9737    0.9367        38
           4     0.9500    0.4872    0.6441        39

    accuracy                         0.8671       173
   macro avg     0.9050    0.8777    0.8696       173
weighted avg     0.8973    0.8671    0.8581       173


Caricamento del modello migliore...

Esecuzione del test sul dataset di test...

TEST REPORT:

              precision    recall  f1-score   support

    chitarra     1.0000    1.0000    1.0000        30
      flauto     1.0000    0.9767    0.9882        43
  pianoforte     0.9655    1.0000    0.9825        28
       viola     0.9744    0.9744    0.9744        39
     violino     0.9750    0.9750    0.9750        40

    accuracy                         0.9833       180
   macro avg     0.9830    0.9852    0.9840       180
weighted avg     0.9835    0.9833    0.9833       180

(base) giovanni02@MacBook-Air-del-Professore Classification-instruments % 
"""