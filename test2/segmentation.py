import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram


def split_audio_to_segments(audio_path, segment_duration=5, output_dir="segments", base_name="audio"):
    """
    Carica un file .wav, lo divide in segmenti di durata fissa (con padding se necessario)
    e salva gli spettrogrammi (in scala di grigi) in una cartella.

    :param audio_path: percorso al file .wav
    :param segment_duration: durata segmenti in secondi (default 5)
    :param output_dir: cartella in cui salvare spettrogrammi
    :param base_name: prefisso per i file salvati
    """
    # Legge il file audio
    sample_rate, data = wavfile.read(audio_path)

    # Se stereo -> prendi solo un canale
    if len(data.shape) == 2:
        data = data[:, 0]

    # Campioni per segmento
    samples_per_segment = segment_duration * sample_rate

    # Crea cartella output
    os.makedirs(output_dir, exist_ok=True)

    num_segments = int(np.ceil(len(data) / samples_per_segment))

    for i in range(num_segments):
        start = i * samples_per_segment
        end = min((i + 1) * samples_per_segment, len(data))
        segment = data[start:end]

        # Padding se segmento troppo corto
        if len(segment) < samples_per_segment:
            padding = np.zeros(samples_per_segment - len(segment), dtype=segment.dtype)
            segment = np.concatenate((segment, padding))

        # Calcola spettrogramma
        f, t, Sxx = spectrogram(segment, fs=sample_rate)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)  # Converti in scala logaritmica per visibilità

        # Salva spettrogramma in scala di grigi
        output_path = os.path.join(output_dir, f"{base_name}_spec_seg{i+1}.png")
        plt.imsave(output_path, Sxx_log, cmap="gray", origin="lower")

        print(f"Salvato: {output_path}")

split_audio_to_segments("./tutte le tracce/viola/Viola 15.wav", segment_duration=5, output_dir="spettrogrammi/viola/traccia15", base_name="traccia15_viola")
