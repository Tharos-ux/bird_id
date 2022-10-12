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

# passer tous les appels à plots dans le thread principal


def futures_collector(func: Callable, argslist: list, num_processes: int) -> list:
    """
    Spawns len(arglist) instances of func and executes them at num_processes instances at time.

    * func : a function
    * argslist (list): a list of tuples, arguments of each func
    * num_processes (int) : max number of concurrent instances
    """
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(func, *args) for args in argslist]
    return [f.result() for f in futures]


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


def audio_processing(data_path: str, output_path: str) -> None:
    """Exports raw audios into pre-processed spectrograms

    Args:
        data_path (str): directory containing species folders
    """
    argslist: list = [[output_path, specie, nb_specie, data_path]
                      for nb_specie, specie in enumerate(listdir(f"{data_path}/"))]
    list_of_l_chunks = futures_collector(
        specie_processing, argslist, cpu_count())
    for i, bird in list_of_l_chunks:
        for j, l_chunks in enumerate(bird):
            # creates spectrogram and exports them in
            export_spectro(l_chunks, argslist[i][1],
                           listdir(f"{data_path}/{argslist[i][1]}/")[j].split('.')[0], output_path)
            # train_set/<specie_name>/<file_ID>_<spec_nbr>.png


def specie_processing(output_path: str, specie: str, nb_specie: int, data_path: str):
    list_of_l_chunks = []
    Path(f"{output_path}/{specie}").mkdir(parents=True, exist_ok=True)
    critical(
        f"Processing specie n°{nb_specie+1}/{len(listdir(data_path+'/'))}")
    for raw_audio in listdir(f"{data_path}/{specie}/"):
        # cut the audio into chunks
        list_of_l_chunks += load_in_blocks(f"{data_path}/{specie}/{raw_audio}")
    return list_of_l_chunks


# @timer("export_spectro")
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


# @timer("load_in_blocks")
def load_in_blocks(audio_path: str, frame_size: int = 5, limit_chunks: int = 30):
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

    """
    # limite de chargement, en nombre de chunks
    l_chunks: list[int] = [0 for _ in range(
        min(int(available_time//frame_size), limit_chunks))]

    for idx, _ in enumerate(l_chunks):
        l_chunks[idx], _ = librosa.core.load(
            audio_path, mono=True, sr=sr, offset=idx*frame_size, duration=frame_size)
    """

    limit: int = min(int(available_time//frame_size), limit_chunks)
    window: int = len(entire_audio)//limit
    return [entire_audio[idx*window:idx*window+window] for idx in range(limit)]
