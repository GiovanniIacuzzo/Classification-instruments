import torch
from tqdm import tqdm

def test_model(model, test_loader, device=None, model_path=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if model_path:
        model.load_state_dict(torch.load(model_path))

    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy, y_true, y_pred
