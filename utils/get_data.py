from utils.dataset import AudioDataset
from torch.utils.data import DataLoader
import torch

def collate_fn(batch):
    waveforms, labels = zip(*batch)
    for i, wf in enumerate(waveforms):
        if wf.shape != waveforms[0].shape:
            print(f"Waveform at index {i} has shape {wf.shape}, expected {waveforms[0].shape}")
    waveforms = torch.stack(waveforms)
    labels = torch.tensor(labels)
    return waveforms, labels


def get_dataloaders(base_path, batch_size=32, num_workers=2, max_len=160000):
    train_set = AudioDataset(base_path, split='train', max_len=max_len)
    val_set = AudioDataset(base_path, split='val', max_len=max_len)
    test_set = AudioDataset(base_path, split='test', max_len=max_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
