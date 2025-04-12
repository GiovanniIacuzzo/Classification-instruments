import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImmaginiDataset(Dataset):
    def __init__(self, data_dir, split="train", img_size=224, transform=None, shuffle=False):
        self.data_dir = os.path.join(data_dir, split)
        self.img_size = img_size

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"La cartella {self.data_dir} non esiste.")

        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.classes = sorted(os.listdir(self.data_dir))

        if not self.classes:
            raise FileNotFoundError(f"Nessuna classe trovata nella cartella {self.data_dir}")

        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name, 'immagini')
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.jpg', '.png')):
                        img_path = os.path.join(class_dir, filename)
                        self.image_paths.append(img_path)
                        self.labels.append(label)

        if shuffle:
            combined = list(zip(self.image_paths, self.labels))
            random.shuffle(combined)
            self.image_paths[:], self.labels[:] = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('L')
        image = self.transform(image)

        return image, label
