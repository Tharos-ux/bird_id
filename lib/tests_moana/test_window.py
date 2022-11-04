# from torchlibrosa.stft import Spectrogram
# https://towardsdatascience.com/sound-event-classification-using-machine-learning-8768092beafc
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import os
from pathlib import Path
import noisereduce as nr


file = "XC51758.mp3"


signal, sr = librosa.load(file, sr=22050)  # can add "mono=False" for stereo

librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

entire_audio, sr = librosa.core.load(
    file, mono=True)  # to get initial audio duration

# CLEAN NOISE FROM ENTIRE AUDIO
noisy_part = signal[0:]
entire_audio = nr.reduce_noise(y=signal, sr=sr)

available_time = librosa.get_duration(entire_audio)
frame_size = 5
limit_chunks = 30
mean_amplitude_entire_audio = np.mean(np.abs(entire_audio))
var_amplitude_entire_audio = abs(np.var(np.abs(entire_audio)))

# limite de chargement, en nombre de chunks
l_chunks: list[int] = [0 for _ in range(
    min(int(available_time//frame_size), limit_chunks))]
mean_amplitude_chunk = [0 for _ in range(
    min(int(available_time//frame_size), limit_chunks))]
mean_amplitude_chunk_norm = [0 for _ in range(
    min(int(available_time//frame_size), limit_chunks))]


for idx, _ in enumerate(l_chunks):
    l_chunks[idx], _ = librosa.core.load(
        file, mono=True, sr=sr, offset=idx*frame_size, duration=frame_size)
    mean_amplitude_chunk[idx] = np.mean(np.abs(l_chunks[idx]))
    mean_amplitude_chunk_norm[idx] = (mean_amplitude_chunk[idx] /
                                      (mean_amplitude_entire_audio + var_amplitude_entire_audio))


for idx_chunk, chunk in enumerate(l_chunks):
    print(mean_amplitude_chunk_norm[idx_chunk])
    if mean_amplitude_chunk_norm[idx_chunk] > 1:
        spectro = librosa.stft(chunk)
        img = librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(spectro), ref=np.max))
        plt.savefig(f"{idx_chunk}.png", bbox_inches="tight", pad_inches=-0.1)


'''
spectrogram_extractor = Spectrogram(
    win_length=1024,
    hop_length=320
).cuda()

# get raw audio data
example, _ = librosa.load(file, sr=32000, mono=True)

raw_audio = torch.Tensor(example).unsqueeze(0).cuda()
spectrogram = spectrogram_extractor(raw_audio)

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

'''
