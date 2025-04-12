import torch
from utils.dataset_audio import Dataset
from models.model_audio import RNN
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

def evaluate_model(test_dir):
    dataset = Dataset(test_dir)
    loader = DataLoader(dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNN(num_classes=len(dataset.label_map)).to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1).transpose(1, 2)
            y = y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print(classification_report(all_labels, all_preds))
