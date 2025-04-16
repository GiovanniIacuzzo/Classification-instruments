import os
import torchaudio
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

FIXED_LENGTH = 48000  # 3 secondi a 16kHz
DOWNSAMPLE_FACTOR = 4

class AudioDataset(Dataset):
    def __init__(self, root_dir, split="train", sample_rate=16000, normalize=True):
        self.root_dir = os.path.join(root_dir, split)
        self.sample_rate = sample_rate
        self.audio_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.normalize = normalize
        self.train = split == "train"

        self._prepare_dataset()

    def _prepare_dataset(self):
        label_set = set()

        for label in os.listdir(self.root_dir):
            label_audio_dir = os.path.join(self.root_dir, label, "audio")
            if not os.path.isdir(label_audio_dir):
                continue

            for filename in os.listdir(label_audio_dir):
                if filename.endswith(".wav"):
                    raw_path = os.path.join(label_audio_dir, filename)
                    self.audio_paths.append(raw_path)
                    self.labels.append(label)
                    label_set.add(label)

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(label_set))}

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        raw_path = self.audio_paths[idx]
        label_str = self.labels[idx]
        label = self.label_to_idx[label_str]

        waveform, sr = torchaudio.load(raw_path)

        if waveform.shape[0] > 1:
            print(f"Convertito in mono: {raw_path}")
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sr != self.sample_rate:
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Normalize
        if self.normalize:
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

        if waveform.shape[1] < FIXED_LENGTH:
            pad_len = FIXED_LENGTH - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :FIXED_LENGTH]

        # Downsampling
        waveform = waveform[:, ::DOWNSAMPLE_FACTOR]

        if self.train:
            waveform = add_noise(waveform)

        return waveform, label

    def get_label_mapping(self):
        return self.label_to_idx

def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def collate_fn(batch):
    waveforms, labels = zip(*batch)
    waveforms = [w.squeeze(0).unsqueeze(-1) for w in waveforms]
    return torch.stack(waveforms), torch.tensor(labels)
