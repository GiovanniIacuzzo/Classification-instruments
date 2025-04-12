import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, data_dir, split="train", sample_rate=16000, duration=3, transform=None, shuffle=False):
        self.data_dir = os.path.join(data_dir, split)
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration
        self.transform = transform

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"La cartella {self.data_dir} non esiste.")

        self.classes = sorted(os.listdir(self.data_dir))
        if not self.classes:
            raise FileNotFoundError(f"Nessuna classe trovata nella cartella {self.data_dir}")

        self.audio_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name, 'audio')
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith('.wav'):
                        path = os.path.join(class_dir, filename)
                        self.audio_paths.append(path)
                        self.labels.append(label)

        if shuffle:
            combined = list(zip(self.audio_paths, self.labels))
            random.shuffle(combined)
            self.audio_paths[:], self.labels[:] = zip(*combined)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)

        # Resample se necessario
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Pad o truncate
        if waveform.size(0) < self.num_samples:
            padding = self.num_samples - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:self.num_samples]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform.unsqueeze(0), label 
