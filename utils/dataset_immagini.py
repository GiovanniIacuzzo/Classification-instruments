import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImmaginiDataset(Dataset):
    def __init__(self, root_dir, subset='train', img_size=224, transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.img_size = img_size

        # Trasformazioni di default per grayscale
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform

        self.image_paths = []
        self.labels = []
        self.classes = []

        # Scansione delle classi (strumenti)
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name, subset)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                for traccia_folder in os.listdir(class_path):
                    traccia_path = os.path.join(class_path, traccia_folder)
                    if os.path.isdir(traccia_path):
                        for img_file in os.listdir(traccia_path):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                                self.image_paths.append(os.path.join(traccia_path, img_file))
                                self.labels.append(class_name)

        # Mappa da classe a indice
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_name = self.labels[idx]
        label_idx = self.class_to_idx[label_name]

        # Carica immagine in grayscale
        image = Image.open(img_path).convert('L')
        image = self.transform(image)

        return image, label_idx
