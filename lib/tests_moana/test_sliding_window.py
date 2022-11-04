# from torchlibrosa.stft import Spectrogram
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import os
from pathlib import Path
import noisereduce as nr
#from scipy.io import wavfile

file = "XC51758.mp3"


signal, sr = librosa.load(file, sr=22050)  # can add "mono=False" for stereo

entire_audio, sr = librosa.core.load(
    file, mono=True)  # to get initial audio duration

# CLEAN NOISE FROM ENTIRE AUDIO
noisy_part = signal[0:]
entire_audio = nr.reduce_noise(y=signal, sr=sr)
#trimmed_entire_audio, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)

librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

librosa.display.waveshow(entire_audio, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

available_time = librosa.get_duration(entire_audio)
frame_size = 5
slide_size = 2
nb_sliding_windows = int(
    (available_time - frame_size + slide_size)//slide_size)
#limit_chunks = 100

# limite de chargement, en nombre de chunks
l_chunks: list[int] = [0 for _ in range(0, nb_sliding_windows)]
mean_amplitude_chunk = [0 for _ in range(0, nb_sliding_windows)]

for idx, _ in enumerate(l_chunks):
    l_chunks[idx], _ = librosa.core.load(
        file, mono=True, sr=sr, offset=idx*slide_size, duration=frame_size)
    mean_amplitude_chunk[idx] = np.mean(np.abs(l_chunks[idx]))
if len(mean_amplitude_chunk) <= 30:
    nb_spectro = len(mean_amplitude_chunk)
else:
    nb_spectro = 30

# GET INDICES OF THE HIGHEST MEANS (nb_spectro biggest chunks values)
print(nb_spectro)
ind = np.argpartition(mean_amplitude_chunk, -nb_spectro)[-nb_spectro:]

for max_id in ind:
    spectro = librosa.stft(l_chunks[max_id])
    img = librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(spectro), ref=np.max))
    plt.savefig(f"{max_id}.png", bbox_inches="tight", pad_inches=-0.1)
