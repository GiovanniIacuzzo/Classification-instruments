import os
import torchaudio
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

class AudioDataset(Dataset):
    def __init__(self, root_dir, split="train", sample_rate=16000, normalize=True):
        self.root_dir = os.path.join(root_dir, split)
        self.sample_rate = sample_rate
        self.audio_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.max_length = 0
        self.normalize = normalize

        self._prepare_dataset()

    def _prepare_dataset(self):
        label_set = set()
        lengths = []

        for label in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label, "audio")
            if not os.path.isdir(label_path):
                continue
            for filename in os.listdir(label_path):
                if filename.endswith(".wav"):
                    full_path = os.path.join(label_path, filename)
                    self.audio_paths.append(full_path)
                    self.labels.append(label)
                    label_set.add(label)

                    info = torchaudio.info(full_path)
                    lengths.append(info.num_frames)

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(label_set))}
        self.max_length = max(lengths)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label_str = self.labels[idx]
        label = self.label_to_idx[label_str]

        waveform, sr = torchaudio.load(path)

        # Convertire in mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resampling
        if sr != self.sample_rate:
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Normalizzazione
        if self.normalize:
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

        return waveform, label

    def get_label_mapping(self):
        return self.label_to_idx

    def get_max_length(self):
        return self.max_length

def collate_fn(batch):
    waveforms, labels = zip(*batch)
    max_len = max(waveform.shape[1] for waveform in waveforms)
    padded_waveforms = []
    
    for waveform in waveforms:
        pad_len = max_len - waveform.shape[1]
        if pad_len > 0:
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        waveform = waveform.squeeze(0).unsqueeze(-1)
        padded_waveforms.append(waveform)

    return torch.stack(padded_waveforms), torch.tensor(labels)

