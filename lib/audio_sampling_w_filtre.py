import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from pathlib import Path
from logging import critical
from time import monotonic
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from multiprocessing import cpu_count
from argparse import ArgumentParser

def timer(arg: str):
    """
    Decorator ; prints out execution time of decorated func
    * arg : descrptor name of job
    """
    def my_inner_dec(func):
        def wrapper(*args, **kwargs):
            print("Starting job...")
            start_time = monotonic()
            res = func(*args, **kwargs)
            end_time = monotonic()
            print(
                f"{arg} : Finished after {timedelta(seconds=end_time - start_time)} seconds")
            return res
        return wrapper
    return my_inner_dec


def audio_processing(data_path: str, output_path: str,specie:str) -> None:
    """Exports raw audios into pre-processed spectrograms

    Args:
        data_path (str): directory containing species folders
    """
    critical(f"Processing specie '{specie}'")
    for raw_audio in listdir(f"{data_path}/{specie}/"):
        # cut the audio into chunks
        l_chunks = load_in_blocks(f"{data_path}/{specie}/{raw_audio}")
        # creates spectrogram and exports them in
        export_spectro(l_chunks, specie,
                        raw_audio.split('.')[0], output_path)


def specie_processing(output_path: str, specie: str, nb_specie: int, data_path: str):
    list_of_l_chunks = []
    Path(f"{output_path}/{specie}").mkdir(parents=True, exist_ok=True)
    critical(
        f"Processing specie n°{nb_specie+1}/{len(listdir(data_path+'/'))}")
    for raw_audio in listdir(f"{data_path}/{specie}/"):
        # cut the audio into chunks
        list_of_l_chunks += load_in_blocks(f"{data_path}/{specie}/{raw_audio}")
    return list_of_l_chunks


def export_spectro(l_chunks: list, specie_name: str, filename: str, output_path: str):
    """ Converts audio into spectros and exports them """
    for idx_chunk, chunk in enumerate(l_chunks):
        spectro = librosa.stft(chunk)
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(spectro), ref=np.max)
        )
        plt.savefig(f"{output_path}/{specie_name}/{filename}_spec_{idx_chunk}.png",
                    bbox_inches="tight", pad_inches=-0.1)
        plt.close()


def load_in_blocks(audio_path: str, frame_size: int = 5, limit_chunks: int = 30, filter:bool = True, overlap: float = 0.5):
    """Chunks audio into parts of 'frame_size' seconds

    Args:
        entry_path (str): path to audio
        frame_size (int, optional): chunks size. Defaults to 5.

    Returns:
        list of chunks
    """
    entire_audio, sr = librosa.core.load(
        audio_path, mono=True, sr=22050, res_type='kaiser_fast')  # to get initial audio duration
    available_time = librosa.get_duration(y=entire_audio)
    limit: int = min(int(available_time//frame_size), limit_chunks)
    window: int = len(entire_audio)//limit
    if filter:
        mean_amplitude_chunk = [0 for _ in range(limit)]
        mean_amplitude_chunk_norm = [0 for _ in range(limit)]
        mean_amplitude_entire_audio = np.mean(np.abs(entire_audio))
        var_amplitude_entire_audio = abs(np.var(np.abs(entire_audio)))
        l_chunks = list()
        for idx in range(limit):
            # j'ai décomposé pour que ce soit plus facile à comprendre/modifier mais on pourra condenser :)
            chunk = entire_audio[idx*window:idx*window+window]
            mean_amplitude_chunk[idx] = np.mean(np.abs(chunk))
            mean_amplitude_chunk_norm[idx] = (mean_amplitude_chunk[idx] /
                                    (mean_amplitude_entire_audio + var_amplitude_entire_audio))
            if mean_amplitude_chunk_norm[idx] > 1:
                l_chunks.append(chunk)
        return l_chunks
    else :
        return [entire_audio[idx*window:idx*window+window] for idx in range(limit)]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="path to a params .json file", type=str)
    parser.add_argument("output", help="path to a params .json file", type=str)
    parser.add_argument("specie", help="path to a params .json file", type=str)
    args = parser.parse_args()
    audio_processing(args.data, args.output, args.specie)
    critical(f"Job {args.specie} ended sucessfully!")