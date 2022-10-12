import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from pathlib import Path
from logging import critical


def audio_processing(data_path: str, output_path: str) -> None:
    """Exports raw audios into pre-processed spectrograms

    Args:
        data_path (str): directory containing species folders
    """
    for nb_specie, specie in enumerate(listdir(f"{data_path}/")):
        critical(
            f"Processing specie n°{nb_specie+1}/{len(listdir(data_path+'/'))}")
        for raw_audio in listdir(f"{data_path}/{specie}/"):
            # cut the audio into chunks
            l_chunks = load_in_blocks(f"{data_path}/{specie}/{raw_audio}")
            # creates spectrogram and exports them in
            export_spectro(l_chunks, specie,
                           raw_audio.split('.')[0], output_path)
            # train_set/<specie_name>/<file_ID>_<spec_nbr>.png


def export_spectro(l_chunks: list, specie_name: str, filename: str, output_path: str):
    """ Converts audio into spectros and exports them """
    Path(f"{output_path}/{specie_name}").mkdir(parents=True, exist_ok=True)

    for idx_chunk, chunk in enumerate(l_chunks):
        spectro = librosa.stft(chunk)
        img = librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(spectro), ref=np.max))
        plt.savefig(f"{output_path}/{specie_name}/{filename}_spec_{idx_chunk}.png",
                    bbox_inches="tight", pad_inches=-0.1)


def load_in_blocks(audio_path: str, frame_size: int = 5, limit_chunks: int = 30):
    """Chunks audio into parts of 'frame_size' seconds

    Args:
        entry_path (str): path to audio
        frame_size (int, optional): chunks size. Defaults to 5.

    Returns:
        list of chunks
    """
    entire_audio, sr = librosa.core.load(
        audio_path, mono=True)  # to get initial audio duration
    available_time = librosa.get_duration(y=entire_audio)
    # limite de chargement, en nombre de chunks
    l_chunks: list[int] = [0 for _ in range(
        min(int(available_time//frame_size), limit_chunks))]

    for idx, _ in enumerate(l_chunks):
        l_chunks[idx], _ = librosa.core.load(
            audio_path, mono=True, sr=sr, offset=idx*frame_size, duration=frame_size)

    return l_chunks
