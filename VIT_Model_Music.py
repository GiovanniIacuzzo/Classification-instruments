from transformers import ViTImageProcessor, ViTModel
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# Estrattore ViT
# --------------------------
class VisionEmbeddings:
    def __init__(self, model_name='google/vit-base-patch16-224', device='mps'):
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    def extract(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # embedding globale
        return embedding.cpu().numpy().squeeze()

# --------------------------
# Dataset
# --------------------------
class VisionTransformerDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None, device='mps'):
        self.root_dir = root_dir
        self.subset = subset
        self.device = device
        self.transform = transform
        self.embeddings_extractor = VisionEmbeddings(device=device)

        self.image_paths = []
        self.labels = []
        self.classes = []

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name, subset)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                for traccia_folder in os.listdir(class_path):
                    traccia_path = os.path.join(class_path, traccia_folder)
                    if os.path.isdir(traccia_path):
                        for img_file in os.listdir(traccia_path):
                            if img_file.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                                self.image_paths.append(os.path.join(traccia_path, img_file))
                                self.labels.append(class_name)

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_idx = self.class_to_idx[self.labels[idx]]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        embedding = self.embeddings_extractor.extract(image)
        return embedding, label_idx

# --------------------------
# Estrazione embeddings in array
# --------------------------
def dataset_to_arrays(dataset):
    X, y = zip(*[dataset[i] for i in range(len(dataset))])
    return np.array(X), np.array(y)

# --------------------------
# Caricamento dataset
# --------------------------
train_dataset = VisionTransformerDataset('./dataset_diviso', subset='train')
val_dataset   = VisionTransformerDataset('./dataset_diviso', subset='val')
test_dataset  = VisionTransformerDataset('./dataset_diviso', subset='test')

X_train, y_train = dataset_to_arrays(train_dataset)
X_val, y_val     = dataset_to_arrays(val_dataset)
X_test, y_test   = dataset_to_arrays(test_dataset)

# --------------------------
# Funzione per Grid Search e valutazione
# --------------------------
def run_model_gridsearch(model, param_grid, X_train, y_train, X_val, y_val, X_test, y_test, classes, name="Model"):
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"\n{name} - Best params:", grid.best_params_)
    
    val_acc = grid.score(X_val, y_val)
    y_pred = grid.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"{name} Validation Accuracy: {val_acc:.4f}")
    print(f"{name} Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{name.lower().replace(' ','_')}_confusion_matrix.png")
    plt.show()
    
    return val_acc, test_acc

# --------------------------
# Random Forest Grid Search
# --------------------------
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_val_acc, rf_test_acc = run_model_gridsearch(RandomForestClassifier(random_state=42), rf_param_grid,
                                               X_train, y_train, X_val, y_val, X_test, y_test,
                                               train_dataset.classes, name="Random Forest")

# --------------------------
# XGBoost Grid Search
# --------------------------
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
xgb_val_acc, xgb_test_acc = run_model_gridsearch(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                                                 xgb_param_grid,
                                                 X_train, y_train, X_val, y_val, X_test, y_test,
                                                 train_dataset.classes, name="XGBoost")

# --------------------------
# Grafico comparativo accuracy
# --------------------------
plt.figure(figsize=(6,4))
models = ['Random Forest', 'XGBoost']
val_acc = [rf_val_acc, xgb_val_acc]
test_acc = [rf_test_acc, xgb_test_acc]

plt.bar(np.arange(len(models)) - 0.15, val_acc, width=0.3, label='Validation Accuracy')
plt.bar(np.arange(len(models)) + 0.15, test_acc, width=0.3, label='Test Accuracy')
plt.xticks(np.arange(len(models)), models)
plt.ylabel("Accuracy")
plt.title("Comparison of Model Accuracies (Grid Search)")
plt.ylim(0,1)
plt.legend()
plt.tight_layout()
plt.savefig("models_accuracy_comparison_gridsearch.png")
plt.show()
