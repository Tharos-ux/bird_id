import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from logging import critical
from time import monotonic
from datetime import timedelta
from argparse import ArgumentParser
from logging import basicConfig, captureWarnings, ERROR
import json

def audio_processing(data_path: str, output_path: str, specie: str, rating_max: float=3) -> None:
    """Exports raw audios into pre-processed spectrograms

    Args:
        data_path (str): directory containing species folders
    """
    critical(f"Processing specie '{specie}'")
    with open("metadata.json") as file:
        metadata: dict = json.load(file)
    for raw_audio in listdir(f"{data_path}/{specie}/"):
        if metadata[raw_audio] >= rating_max:
            # cut the audio into chunks
            l_chunks = load_in_blocks(f"{data_path}/{specie}/{raw_audio}")
            # creates spectrogram and exports them in
            export_spectro(l_chunks, specie,
                        raw_audio.split('.')[0], output_path)


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
        plt.close()


def load_in_blocks(audio_path: str, frame_size: int = 5, limit_chunks: int = 30, overlap: float = 0.5):
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

    return [entire_audio[int(idx*overlap*window):idx*window+window] for idx in range(limit)]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="path to a params .json file", type=str)
    parser.add_argument("output", help="path to a params .json file", type=str)
    parser.add_argument("specie", help="path to a params .json file", type=str)
    parser.add_argument("-f", "--filter",
                        help="whouhou un filtre", action='store_true')
    args = parser.parse_args()
    captureWarnings(capture=True)
    basicConfig(format='%(asctime)s %(message)s', datefmt='[%m/%d/%Y %I:%M:%S %p]', filename="bird_id.log",
                encoding='utf-8', level=ERROR)
    audio_processing(args.data, args.output, args.specie)
    critical("ça s'est bien passé !")
