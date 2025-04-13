import torchaudio
import torch

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
