from random import random
from time import sleep
from multiprocessing import cpu_count
from psutil import virtual_memory
from json import load
from logging import basicConfig, captureWarnings, ERROR
from argparse import ArgumentParser
from logging import critical
from os import listdir
import gc
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


def audio_processing(data_path: str, output_path: str, specie: str, rating_max: float = 4, filter: bool = False) -> None:
    """Exports raw audios into pre-processed spectrograms

    Args:
        data_path (str): directory containing species folders
    """

    critical(f"Processing specie '{specie}'")
    with open("rating.json") as file:
        rating: dict = load(file)
    count = 0
    for raw_audio in listdir(f"{data_path}/{specie}/"):
        processed: bool = False
        while not processed:
            if rating[raw_audio] >= rating_max:
                try:
                    # cut the audio into chunks
                    l_chunks = load_in_blocks(
                        f"{data_path}/{specie}/{raw_audio}", filter=filter)
                    # creates spectrogram and exports them in
                    export_spectro(l_chunks, specie,
                                   raw_audio.split('.')[0], output_path)
                    processed = True
                except ZeroDivisionError:
                    processed = True
                except Exception as exc:
                    next_iter: float = 10 + 20*random()
                    critical(
                        f"W : Occured exception on file {raw_audio}.\n                         {exc}\n                         Resuming task in {int(next_iter)} seconds.")
                    sleep(20)
            else:
                processed = True


def export_spectro(l_chunks: list, specie_name: str, filename: str, output_path: str):
    """ Converts audio into spectros and exports them
        /!\ SPECTROS ARE 500x400px for consistency issues --> some weren't this size without fixed params
    Args:
        l_chunks (list): list of chunks
        specie_name (str): _description_
        filename (str): _description_
        output_path (str): _description_
    """
    for idx_chunk, chunk in enumerate(l_chunks):
        plt.rcParams["figure.figsize"] = (5, 4)
        spectro = librosa.stft(chunk)
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(spectro), ref=np.max)
        )
        plt.savefig(f"{output_path}/{specie_name}/{filename}_spec_{idx_chunk}.png",
                    bbox_inches="tight", pad_inches=-0.1, dpi=100)
        plt.close("all")
        plt.close()


def load_in_blocks(audio_path: str, frame_size: int = 3, limit_chunks: int = 30,
                   filter: bool = False, overlap: float = 0.5):
    """Chunks audio into parts of 'frame_size' seconds

    Args:
        entry_path (str): path to audio
        frame_size (int, optional): chunks size. Defaults to 5.
        overlap (float, optional): overlap percentage
    Returns:
        list of chunks
    """
    entire_audio, sr = librosa.core.load(
        audio_path, mono=True, sr=22050, res_type='kaiser_fast')  # to get initial audio duration
    available_time = librosa.get_duration(y=entire_audio)

    limit: int = min(int(available_time//frame_size), limit_chunks)
    if limit == 0:
        limit = limit_chunks
    window: int = len(entire_audio)//limit

    if filter:
        mean_amplitude_chunk = [0 for _ in range(limit)]
        mean_amplitude_chunk_norm = [0 for _ in range(limit)]
        mean_amplitude_entire_audio = np.mean(np.abs(entire_audio))
        var_amplitude_entire_audio = abs(np.var(np.abs(entire_audio)))
        l_chunks = list()
        for idx in range(limit):
            # j'ai décomposé pour que ce soit plus facile à comprendre/modifier mais on pourra condenser :)
            chunk = entire_audio[int(idx*window*overlap):idx*window + window]
            mean_amplitude_chunk[idx] = np.mean(np.abs(chunk))
            mean_amplitude_chunk_norm[idx] = (mean_amplitude_chunk[idx] /
                                              (mean_amplitude_entire_audio + var_amplitude_entire_audio))
            if mean_amplitude_chunk_norm[idx] > 1:
                l_chunks.append(chunk)
        return l_chunks
    else:
        return [entire_audio[int(idx*overlap*window):idx*window + window] for idx in range(limit)]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="path to a params .json file", type=str)
    parser.add_argument("output", help="path to a params .json file", type=str)
    parser.add_argument("specie", help="path to a params .json file", type=str)
    parser.add_argument("max_spectro",
                        help="specifies max spectro exported for one specie", type=int)
    parser.add_argument("-f", "--filter",
                        help="whouhou un filtre", action='store_true')

    args = parser.parse_args()
    captureWarnings(capture=True)
    basicConfig(format='%(asctime)s %(message)s', datefmt='[%m/%d/%Y %I:%M:%S %p]', filename="bird_id.log",
                encoding='utf-8', level=ERROR)
    if args.filter:
        audio_processing(args.data, args.output, args.specie,
                         args.max_spectro, filter=True)
    else:
        audio_processing(args.data, args.output, args.specie)
    critical(f"Job {args.specie} ended sucessfully!")
