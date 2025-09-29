from transformers import AutoFeatureExtractor, AutoModel
import torch
from torch.utils.data import Dataset
import torchaudio
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# La tua classe AudioEmbeddings rimane intatta
# --------------------------
class AudioEmbeddings:
    '''
    This class is intended to extract embeddings from audio models.
    It uses Wav2Vec2 as a default model.
    '''
    
    def __init__(self, model_name='ALM/hubert-base-audioset', device='cuda'):
        
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = device
        self.model.to(self.device)
        
        self.model_name = model_name
        
        # eval mode
        self.model.eval()
        
    def extract(self, speech):
        '''
        Extract embeddings from a speech.
        
        Args:
            speech (torch.Tensor or np.ndarray): 1D audio array
        
        Returns:
            np.ndarray: Embeddings of the speech
        '''
        
        # Assicuriamoci che sia numpy
        if isinstance(speech, torch.Tensor):
            speech = speech.cpu().numpy()
        
        inputs = self.processor(speech, return_tensors="pt", padding="longest")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Media sui frame → vettore 1D
        embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().flatten()
        return embedding.astype(np.float32)


# --------------------------
# Dataset
# --------------------------
class AudioDataset(Dataset):
    def __init__(self, root_dir, subset='train', device='cpu', save_dir="embeddings"):
        self.root_dir = root_dir
        self.subset = subset
        self.device = device
        self.embeddings_extractor = AudioEmbeddings(device=device)
        self.save_dir = os.path.join(save_dir, subset)
        os.makedirs(self.save_dir, exist_ok=True)

        self.audio_paths = []
        self.labels = []
        self.classes = []

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name, subset)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                for wav_file in os.listdir(class_path):
                    if wav_file.lower().endswith('.wav'):
                        self.audio_paths.append(os.path.join(class_path, wav_file))
                        self.labels.append(class_name)

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        print(f"[DEBUG] {subset} set: trovati {len(self.audio_paths)} file audio.")

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label_idx = self.class_to_idx[self.labels[idx]]

        # file per cache embeddings
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        emb_path = os.path.join(self.save_dir, f"{base_name}.npy")

        if os.path.exists(emb_path):
            embedding = np.load(emb_path)
        else:
            waveform, sr = torchaudio.load(audio_path)  # waveform: [channels, length]

            # Se stereo, usa solo il primo canale
            if waveform.shape[0] > 1:
                waveform = waveform[0:1, :]
            waveform = waveform.squeeze(0)  # ora [length]

            # Resample al sampling rate del modello
            target_sr = self.embeddings_extractor.processor.sampling_rate
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)

            embedding = self.embeddings_extractor.extract(waveform)
            np.save(emb_path, embedding)

        return embedding, label_idx


# --------------------------
# Estrazione embeddings in array con padding
# --------------------------
def dataset_to_arrays(dataset):
    X, y = [], []
    dims = []

    # Primo passaggio: raccogli tutte le embedding e le loro dimensioni
    for i in range(len(dataset)):
        embedding, label = dataset[i]
        X.append(embedding)
        y.append(label)
        dims.append(embedding.shape[0])

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if (i + 1) % 5 == 0:
            print(f"[DEBUG] Processati {i+1}/{len(dataset)} file...")

    # Trova la dimensione massima
    max_dim = max(dims)
    print(f"[DEBUG] Dimensione massima embedding trovata: {max_dim}")

    # Padding per uniformare
    X_padded = []
    for emb in X:
        if emb.shape[0] < max_dim:
            padded = np.zeros(max_dim, dtype=np.float32)
            padded[:emb.shape[0]] = emb
            X_padded.append(padded)
        else:
            X_padded.append(emb)

    return np.vstack(X_padded), np.array(y)


# --------------------------
# Caricamento dataset
# --------------------------
train_dataset = AudioDataset('./tutte_le_tracce', subset='train', device="cpu")
val_dataset   = AudioDataset('./tutte_le_tracce', subset='val', device="cpu")
test_dataset  = AudioDataset('./tutte_le_tracce', subset='test', device="cpu")

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
    
    print(classification_report(y_test, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{name.lower().replace(' ','_')}_confusion_matrix.png")
    plt.close()
    
    return val_acc, test_acc


# --------------------------
# Random Forest Grid Search
# --------------------------
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_val_acc, rf_test_acc = run_model_gridsearch(
    RandomForestClassifier(random_state=42), rf_param_grid,
    X_train, y_train, X_val, y_val, X_test, y_test,
    train_dataset.classes, name="Random Forest"
)


# --------------------------
# XGBoost Grid Search
# --------------------------
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}
xgb_val_acc, xgb_test_acc = run_model_gridsearch(
    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    xgb_param_grid,
    X_train, y_train, X_val, y_val, X_test, y_test,
    train_dataset.classes, name="XGBoost"
)


# --------------------------
# Grafico comparativo finale
# --------------------------
models = ['Random Forest', 'XGBoost']
val_accs = [rf_val_acc, xgb_val_acc]
test_accs = [rf_test_acc, xgb_test_acc]

x = np.arange(len(models))

plt.figure(figsize=(8,6))
plt.bar(x - 0.2, val_accs, width=0.4, label='Validation Accuracy')
plt.bar(x + 0.2, test_accs, width=0.4, label='Test Accuracy')

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Confronto modelli Audio - Validation vs Test")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("audio_models_accuracy_comparison.png")
plt.show()

"""
(base) giovanni02@MacBook-Air-del-Professore Classification-instruments % /usr/local/bin/python3 /Users/giovanni02/Desktop/Progetti/Classification-instruments/AudioTrasformers.py
[DEBUG] train set: trovati 24 file audio.
[DEBUG] val set: trovati 11 file audio.
[DEBUG] test set: trovati 12 file audio.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Processati 5/24 file...
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Processati 10/24 file...
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Processati 15/24 file...
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Processati 20/24 file...
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Dimensione massima embedding trovata: 768
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Processati 5/11 file...
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Processati 10/11 file...
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Dimensione massima embedding trovata: 768
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Processati 5/12 file...
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Processati 10/12 file...
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
It is strongly recommended to pass the `sampling_rate` argument to `Wav2Vec2FeatureExtractor()`. Failing to do so can result in silent errors that might be hard to debug.
[DEBUG] Dimensione massima embedding trovata: 768
Python(68558) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn/model_selection/_split.py:776: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=3.
  warnings.warn(
Python(68559) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(68560) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(68561) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(68562) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(68563) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(68564) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(68565) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(68566) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(68567) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.

Random Forest - Best params: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
              precision    recall  f1-score   support

    chitarra       1.00      1.00      1.00         2
      flauto       1.00      1.00      1.00         2
  pianoforte       1.00      1.00      1.00         1
       viola       0.80      1.00      0.89         4
     violino       1.00      0.67      0.80         3

    accuracy                           0.92        12
   macro avg       0.96      0.93      0.94        12
weighted avg       0.93      0.92      0.91        12

/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn/model_selection/_split.py:776: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=3.
  warnings.warn(

XGBoost - Best params: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
              precision    recall  f1-score   support

    chitarra       0.67      1.00      0.80         2
      flauto       0.50      0.50      0.50         2
  pianoforte       0.00      0.00      0.00         1
       viola       0.67      1.00      0.80         4
     violino       1.00      0.33      0.50         3

    accuracy                           0.67        12
   macro avg       0.57      0.57      0.52        12
weighted avg       0.67      0.67      0.61        12

(base) giovanni02@MacBook-Air-del-Professore Classification-instruments % 
"""