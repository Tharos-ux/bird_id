from lib import audio_processing
from constants import PATH_DATA, PATH_TRAIN
from argparse import ArgumentParser
from os import system
from logging import basicConfig, captureWarnings, INFO, CRITICAL


def setup_logs() -> None:
    # capturing PySoundFile warnings and filtering them
    with open("bird_id.log", 'w'):
        pass
    captureWarnings(capture=True)
    basicConfig(format='%(asctime)s %(message)s', datefmt='[%m/%d/%Y %I:%M:%S %p]', filename="bird_id.log",
                encoding='utf-8', level=CRITICAL)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--clean",
                        help="Erase all data inside training folder", action='store_true')
    parser.add_argument("-s", "--spectrograms",
                        help="Builds the spectrograms from the current data folder", action='store_true')
    args = parser.parse_args()

    setup_logs()

    # clean whole train set from disk
    if args.clean:
        system(f"rm -r {PATH_TRAIN}/")

    # plotting spectrograms for whole data folder
    if args.spectrograms:
        audio_processing(data_path=PATH_DATA, output_path=PATH_TRAIN)
