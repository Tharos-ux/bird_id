from glob import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def audio_processing(dir):
    """ Exports raw audios into pre-processed spectrograms """

    l_raw_audios = glob(dir)

    for raw_audio in l_raw_audios:
        l_chunks = load_in_blocks(raw_audio) # cut the audio into chunks
        
        dir = raw_audio[21:27] # get specie name
        filename_init = raw_audio[28:-4] # get file ID
        export_spectro(l_chunks, dir, filename_init) # creates spectrogram and exports them in
                                             # train_set/<specie_name>/<file_ID>_<spec_nbr>.png

def export_spectro(l_chunks:list, dir:str, filename_init:str):
    """ Converts audio into spectros and exports them """

    try:
        os.chdir('train_set/' + dir)
    except:
        os.mkdir('train_set/' + dir) # create a new directory to store spectro
        os.chdir('train_set/' + dir)

    for idx_chunk, chunk in enumerate(l_chunks):
        spectro = librosa.stft(chunk)

        img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(spectro), ref=np.max))
        path = filename + '_spec_' + str(idx_chunk) + '.png'
        plt.savefig(path)
    os.chdir('../..')


def load_in_blocks(audio_path:str, frame_size: int=5):
    """Chunks audio into parts of 'frame_size' seconds

    Args:
        entry_path (str): path to audio
        frame_size (int, optional): chunks size. Defaults to 5.

    Returns:
        list of chunks
    """
    entire_audio, sr = librosa.core.load(audio_path, mono=True) # to get initial audio duration
    available_time = librosa.get_duration(entire_audio)
    l_chunks = [0]*int(available_time//frame_size)

    offset = 0 # start reading after this time (in seconds)
    idx = 0
    # While audio isn't finished
    while available_time > frame_size:
        chunk, sr = librosa.core.load(audio_path, mono=True sr=sr, offset=offset, duration=frame_size)
        l_chunks[idx] = chunk

        offset += frame_size
        available_time -= frame_size
        idx += 1

    return l_chunks

dir = 'birdsong-recognition/brdowl/*'
list = audio_processing(dir)
