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


"""
(base) giovanni02@MacBook-Air-del-Professore Classification-instruments % /usr/local/bin/python3 /Users/giovanni02/Desktop/Progetti/Classifi
cation-instruments/VIT_Model_Music.py
Some weights of ViTModel were not initialized from the model checkpoint at google/vit-base-patch16-224 and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of ViTModel were not initialized from the model checkpoint at google/vit-base-patch16-224 and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of ViTModel were not initialized from the model checkpoint at google/vit-base-patch16-224 and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

Random Forest - Best params: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 300}
Random Forest Validation Accuracy: 0.7168
Random Forest Test Accuracy: 0.8222
              precision    recall  f1-score   support

    chitarra       0.79      0.90      0.84        30
      flauto       0.89      0.91      0.90        43
  pianoforte       0.93      1.00      0.97        28
       viola       0.72      0.85      0.78        39
     violino       0.81      0.53      0.64        40

    accuracy                           0.82       180
   macro avg       0.83      0.84      0.82       180
weighted avg       0.82      0.82      0.81       180


XGBoost - Best params: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'subsample': 1.0}
XGBoost Validation Accuracy: 0.7746
XGBoost Test Accuracy: 0.8278
              precision    recall  f1-score   support

    chitarra       0.84      0.90      0.87        30
      flauto       0.91      0.93      0.92        43
  pianoforte       0.90      1.00      0.95        28
       viola       0.71      0.82      0.76        39
     violino       0.79      0.55      0.65        40

    accuracy                           0.83       180
   macro avg       0.83      0.84      0.83       180
weighted avg       0.83      0.83      0.82       180

(base) giovanni02@MacBook-Air-del-Professore Classification-instruments % 
"""