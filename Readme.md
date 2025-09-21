# 🎵 Classification of Musical Instruments

<p align="center">
  This is the **Italian version by default**. Switch to:  
  <a href="Readme_En.md">🇬🇧 English</a>
</p>

Questo progetto si occupa della classificazione automatica di strumenti musicali a partire da dati audio trasformati in immagini (spettrogrammi). Il problema è affrontato come una **classificazione multilabel**, confrontando le prestazioni di una **CNN (modello di deep learning)** con modelli tradizionali di machine learning: **XGBoost** e **Random Forest**.

---

## 📁 Struttura del Dataset

Il dataset è stato **raccolto manualmente** e comprende **registrazioni audio di 5 strumenti musicali**:

- Chitarra
- Pianoforte
- Violino
- Viola
- Flauto

I file `.wav` sono stati convertiti in spettrogrammi grigi con la libreria `librosa`.

Ogni spettrogramma è poi salvato come immagine `.png` e suddiviso in:

- `train` (70%)
- `val` (15%)
- `test` (15%)

---

Organizzati nella seguente struttura:

```bash
data/
├── train/
│   └── <strumento>/immagini/*.png
├── val/
│   └── <strumento>/immagini/*.png
├── test/
│   └── <strumento>/immagini/*.png
│
├── models/
│   └── model_immagini.py
│
├── pre data/
│   ├── audio.py
│   ├── clone.ipynb
│   └── segmentaion.py
│
├── utils/
│   ├── dataset_immagini.py
│   ├── evaluate_immagini.py
│   ├── train_immagini.py
│   └── test_immagini.py
│
├── main_immagini.ipynb
├── main.ipynb
│
├── extract_features.ipynb
├── ispezione.ipynb
│
├── requirement.txt
├── environment.yml
│
└── Readme.md
```

---

## 🔧 Preprocessing: Da Audio a Spettrogrammi

Il preprocessing converte i file `.wav` in spettrogrammi tramite:

```bash
librosa.feature.melspectrogram()
librosa.power_to_db()
```
I file risultanti vengono salvati come immagini monocromatiche.

---

## 🧠 Modelli Utilizzati

Nel nostro progetto di classificazione multilabel degli strumenti musicali a partire da immagini spettrogrammi, abbiamo adottato e confrontato tre approcci differenti per valutare l’efficacia di modelli basati su deep learning rispetto a metodi tradizionali di machine learning.

### 1. Convolutional Neural Network (CNN)

Abbiamo sviluppato un modello CNN personalizzato utilizzando PyTorch. Il modello è composto da tre blocchi convoluzionali con Batch Normalization, MaxPooling e Dropout per prevenire l'overfitting. La rete termina con due layer fully connected.

- **Input:** spettrogrammi in scala di grigi (1 x 224 x 224)  
- **Output:** probabilità per ciascuna delle 5 classi (strumenti)  
- **Funzione di perdita:** `CrossEntropyLoss`  
- **Ottimizzatore:** Adam con learning rate di 0.001  
- **Early Stopping:** monitorato sulla *validation accuracy* con `patience = 5`

Durante l'addestramento, salviamo il modello con la migliore accuracy sulla validation e generiamo:

- Curva di loss e accuracy per training e validation  
- Matrice di confusione finale  
- Classificazione dettagliata per classe

---

### 2. XGBoost (XGBClassifier)

Come approccio alternativo, abbiamo estratto feature statistiche dagli spettrogrammi (es. media, deviazione standard, skewness) e le abbiamo utilizzate per addestrare un classificatore `XGBClassifier`.

- **Modello:** Gradient Boosting (XGBoost)  
- **Vantaggi:** veloce da addestrare, interpretabilità delle feature  
- **Limiti:** richiede estrazione manuale delle caratteristiche e non sfrutta pienamente la struttura spaziale dell’immagine

---

### 3. Random Forest

Abbiamo infine testato un classificatore Random Forest, anch’esso basato su feature estratte manualmente dagli spettrogrammi. È stato utilizzato come baseline classico:

- **Modello:** ensemble di alberi decisionali  
- **Punti di forza:** robustezza a overfitting e facilità di interpretazione  
- **Limiti:** prestazioni inferiori rispetto alla CNN

---

## 🔍 Confronto

| Modello         | Feature Input            | Approccio        | Vantaggio Principale                         | Accuracy                     |
|-----------------|--------------------------|------------------|----------------------------------------------|------------------------------|
| **CNN**         | Immagini (spettrogrammi) | Deep Learning    | Apprendimento automatico dalle immagini      | Alta (da valutazione finale) |
| **XGBoost**     | Feature estratte         | Machine Learning | Ottima performance su feature numeriche      | Media                        |
| **Random Forest** | Feature estratte       | Machine Learning | Semplice e interpretabile                    | Più bassa                    |

---

## 🛠 Requisiti
```bash
torch
torchvision
matplotlib
seaborn
pandas
scikit-learn
librosa
```
Puoi installarli con:
```bash
pip install -r requirements.txt
```
Oppure puoi configurare un ambiente conda con:
```bash
conda env create -f environment.yml
conda activate classification_instruments
```

## 📎 Licenza
da inserire

---

### 📩 Contatti
Per qualsiasi domanda o richiesta di chiarimento, non esitare a contattarci:

- 📧 [Giovanni Giuseppe Iacuzzo](mailto:giovanni.iacuzzo@unikorestudent.it)  
- 📧 [Chiara Maria Milazzo](mailto:chiara.milazzo@unikorestudent.it)

