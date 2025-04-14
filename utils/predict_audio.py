import torchaudio
import torch
import matplotlib.pyplot as plt

def predict_single_file(model, filepath, label_mapping, device=None, sample_rate=16000, model_path="best_model.pth"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    waveform, sr = torchaudio.load(filepath)

    # Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Normalizza
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

    waveform = waveform.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(waveform)
        predicted_idx = torch.argmax(output, dim=1).item()

    idx_to_label = {v: k for k, v in label_mapping.items()}
    predicted_label = idx_to_label[predicted_idx]
    print(f"Predetto: {predicted_label}")
    return predicted_label


def log_results(log_path, **kwargs):
    with open(log_path, "a") as f:
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
        f.write("-" * 40 + "\n")

def plot_metrics(train_loss, val_loss, train_acc, val_acc, save_dir="plots"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss durante l'allenamento")
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuratezza durante l'allenamento")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.show()

