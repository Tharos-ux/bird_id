from glob import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import os
from pathlib import Path

def audio_processing(data_path:str) -> None:
    """Exports raw audios into pre-processed spectrograms

    Args:
        data_path (str): directory containing species folders
    """
    for specie in listdir(f"{data_path}/"):
        for raw_audio in listdir(f"{data_path}/{specie}/"):
            # cut the audio into chunks
            l_chunks = load_in_blocks(f"{data_path}/{specie}/{raw_audio}")
            # creates spectrogram and exports them in
            export_spectro(l_chunks, specie, raw_audio.split('.')[0])
            # train_set/<specie_name>/<file_ID>_<spec_nbr>.png

def export_spectro(l_chunks:list, specie_name:str, filename:str):
    """ Converts audio into spectros and exports them """
    Path(f"train_set/{specie_name}").mkdir(parents=True, exist_ok=True)

    for idx_chunk, chunk in enumerate(l_chunks):
        spectro = librosa.stft(chunk)
        img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(spectro), ref=np.max))
        plt.savefig(f"train_set/{specie_name}/{filename}_spec_{idx_chunk}.png",bbox_inches="tight",pad_inches=-0.1)


def load_in_blocks(audio_path:str, frame_size: int=5, limit_chunks:int = 30):
    """Chunks audio into parts of 'frame_size' seconds

    Args:
        entry_path (str): path to audio
        frame_size (int, optional): chunks size. Defaults to 5.

    Returns:
        list of chunks
    """
    entire_audio, sr = librosa.core.load(audio_path, mono=True) # to get initial audio duration
    available_time = librosa.get_duration(entire_audio)
    l_chunks:list[int] = [0 for _ in range(min(int(available_time//frame_size),limit_chunks))] # limite de chargement, en nombre de chunks

    for idx,_ in enumerate(l_chunks):
        l_chunks[idx],_ = librosa.core.load(audio_path, mono=True, sr=sr, offset=idx*frame_size, duration=frame_size)

    return l_chunks

if __name__ == "__main__"
    audio_processing(f"birdsong-recognition")
