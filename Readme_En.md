# 🎵 Classification of Musical Instruments

<p align="center">
  Questa è la **Versione in Inglese**. passa a:  
  <a href="README.md">🇮🇹 Italiano</a>
</p>

This project deals with the automatic classification of musical instruments from audio data transformed into images (spectrograms). The problem is approached as a **multilabel classification**, comparing the performance of a **CNN (deep learning model)** with traditional machine learning models: **XGBoost** and **Random Forest**.

---

## 📁 Dataset Structure

The dataset was **manually collected** and includes **audio recordings of 5 musical instruments**:

- Chitarra
- Pianoforte
- Violino
- Viola
- Flauto

The `.wav` files were converted to gray spectrograms using the `librosa` library.

Each spectrogram was then saved as a `.png` image and divided into:

- `train` (70%)
- `val` (15%)
- `test` (15%)

---

Organized in the following structure:

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

## 🔧 Preprocessing: From Audio to Spectrograms

Preprocessing converts `.wav` files to spectrograms via:

```bash
librosa.feature.melspectrogram()
librosa.power_to_db()
```
The resulting files are saved as monochrome images.

---

## 🧠 Models Used

In our multilabel classification project of musical instruments based on spectrogram images, we adopted and compared three different approaches to evaluate the effectiveness of deep learning models against traditional machine learning methods.

### 1. Convolutional Neural Network (CNN)

We developed a custom CNN model using PyTorch. The model consists of three convolutional blocks with Batch Normalization, MaxPooling, and Dropout to prevent overfitting. The network ends with two fully connected layers.

- **Input:** grayscale spectrograms (1 x 224 x 224)  
- **Output:** probability for each of the 5 instrument classes  
- **Loss function:** `CrossEntropyLoss`  
- **Optimizer:** Adam with a learning rate of 0.001  
- **Early Stopping:** monitored on *validation accuracy* with `patience = 5`

During training, we save the model with the best validation accuracy and generate:

- Training and validation loss/accuracy curves  
- Final confusion matrix  
- Detailed classification report per class  

---

### 2. XGBoost (XGBClassifier)

As an alternative approach, we extracted statistical features from the spectrograms (e.g., mean, standard deviation, skewness) and used them to train an `XGBClassifier`.

- **Model:** Gradient Boosting (XGBoost)  
- **Advantages:** fast to train, feature interpretability  
- **Limitations:** requires manual feature extraction and does not fully exploit the spatial structure of the image  

---

### 3. Random Forest

Finally, we tested a Random Forest classifier, also based on manually extracted features from spectrograms. It was used as a classic baseline:

- **Model:** ensemble of decision trees  
- **Strengths:** robustness to overfitting and ease of interpretation  
- **Limitations:** lower performance compared to the CNN  

---

## 🔍 Comparison

| Model           | Feature Input            | Approach         | Main Advantage                               | Accuracy                     |
|-----------------|--------------------------|------------------|----------------------------------------------|------------------------------|
| **CNN**         | Images (spectrograms)    | Deep Learning    | Automatic learning from images               | High (based on final eval.)  |
| **XGBoost**     | Extracted features       | Machine Learning | Great performance on numerical features      | Medium                       |
| **Random Forest** | Extracted features     | Machine Learning | Simple and interpretable                     | Lower                        |

## 🛠 Requirements

```bash
torch
torchvision
matplotlib
seaborn
pandas
scikit-learn
librosa
```
You can install them with:
```bash
pip install -r requirements.txt
```
Or you can set up a conda environment with:
```bash
conda env create -f environment.yml
conda activate classification_instruments
```

## 📎 License
to be inserted

### 📩 Contacts
For any questions or requests for clarification, please don't hesitate to contact us:

- 📧 [Giovanni Giuseppe Iacuzzo](mailto:giovanni.iacuzzo@unikorestudent.it)  
- 📧 [Chiara Maria Milazzo](mailto:chiara.milazzo@unikorestudent.it)