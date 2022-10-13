from lib import audio_processing
from constants import PATH_DATA, PATH_TRAIN
from argparse import ArgumentParser
from os import system
from logging import basicConfig, captureWarnings, ERROR
import shlex
import subprocess


def setup_logs() -> None:
    # capturing PySoundFile warnings and filtering them
    with open("bird_id.log", 'w'):
        pass
    captureWarnings(capture=True)
    basicConfig(format='%(asctime)s %(message)s', datefmt='[%m/%d/%Y %I:%M:%S %p]', filename="bird_id.log",
                encoding='utf-8', level=ERROR)

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
        retlist: list = futures_collector(subprocess.Popen, [[shlex.split(f"python lib/audio_sampling.py {PATH_DATA} {PATH_TRAIN} {specie}")] for specie in listdir(f"{PATH_DATA}/")], cpu_count())
