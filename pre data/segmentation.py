from pydub import AudioSegment
import os

def split_audio(file_path, output_folder, num_segments):
    audio = AudioSegment.from_wav(file_path)
    total_duration_ms = len(audio)

    segment_length_ms = total_duration_ms // num_segments

    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_segments):
        start_time = i * segment_length_ms
        end_time = start_time + segment_length_ms

        if i == num_segments - 1:
            end_time = total_duration_ms

        segment = audio[start_time:end_time]
        output_filename = f'segment_{i+1}.wav'
        segment.export(os.path.join(output_folder, output_filename), format='wav')
        print(f"Segmento {i+1} salvato: {output_filename}")

file_path = "./VIOLINO/VIOLINO.wav"
output_folder = "VIOLINO_tagli"
num_segments = 120 

split_audio(file_path, output_folder, num_segments)
