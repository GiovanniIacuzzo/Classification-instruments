import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

LABELS = ["chitarra", "flauto", "pianoforte", "viola", "violino"]
LABEL2IDX = {label: idx for idx, label in enumerate(LABELS)}

class Dataset(Dataset):
    def __init__(self, root_dir, max_len=160000):
        self.samples = []
        self.root_dir = Path(root_dir)
        self.max_len = max_len

        for label in LABELS:
            label_dir = self.root_dir / label / "audio"
            if not label_dir.exists():
                continue
            for file in label_dir.glob("*.wav"):
                self.samples.append((file, LABEL2IDX[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = self._preprocess(waveform, sample_rate)
        return waveform, label

    def _preprocess(self, waveform, sample_rate, target_sr=16000):
        if sample_rate != target_sr:
            resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resample(waveform)

        # Padding o taglio
        length = waveform.shape[1]
        if length > self.max_len:
            waveform = waveform[:, :self.max_len]
        elif length < self.max_len:
            pad = self.max_len - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        
        return waveform
def get_dataloaders(base_path, batch_size=16, num_workers=2, max_len=160000):
    train_set = Dataset(os.path.join(base_path, "train"), max_len=max_len)
    val_set = Dataset(os.path.join(base_path, "val"), max_len=max_len)
    test_set = Dataset(os.path.join(base_path, "test"), max_len=max_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
