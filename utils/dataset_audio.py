import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

LABELS = ["chitarra", "flauto", "pianoforte", "viola", "violino"]
LABEL2IDX = {label: idx for idx, label in enumerate(LABELS)}

class AudioDataset(Dataset):
    def __init__(self, root_dir, split='train', max_len=160000, target_sr=16000):
        self.samples = []
        self.max_len = max_len
        self.target_sr = target_sr
        self.root_dir = Path(root_dir) / split

        for label in LABELS:
            audio_dir = self.root_dir / label / "audio"
            for file_path in audio_dir.glob("*.wav"):
                self.samples.append((file_path, LABEL2IDX[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # Converti a mono se necessario
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # (1, T)

        # Resampling
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Padding o truncating
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        else:
            pad_length = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        return waveform, label
