import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def wav_to_spectrogram(input_folder, output_folder, nome_base=" "):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    count = 1  # Contatore per numerare i file

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(input_folder, file)
            y, sr = librosa.load(file_path, sr=None)
            
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, cmap='gray_r')
            plt.axis('off')
            
            output_file = os.path.join(output_folder, f"{nome_base}_{count}.png")
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            print(f"Salvato: {output_file}")
            count += 1

input_folder = "./VIOLINO_tagli"
output_folder = "Spettrogrammi"
wav_to_spectrogram(input_folder, output_folder, nome_base="violino")
