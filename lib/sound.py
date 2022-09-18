from math import ceil, floor
from numpy import abs, clip, fft
from mutagen.wave import WAVE
from sounddevice import query_devices, play, InputStream
from soundfile import read


def pre_processing(sound_files: list[str], chan_id: int, width_plage: int, gain: int, lower_bound: int, upper_bound: int, number_of_blocks: int) -> None:
    """Processes spectrograms for all soundclips listed

    Args:
        sound_files (list[str]): list of files to process
        chan_id (int): read channel to listen to (e.g. the one where they are played)
        width_plage (int): number of discrete values in spectrogram
        gain (int): multiplicator (for lisibility purposes)
        lower_bound (int): lower frequency we listen to
        upper_bound (int): higher frequency we listen to
        number_of_blocks (int): number of spectrograms per file
    """
    # Some vars for rendering in console
    usage_line: str = ' press <enter> to quit, +<enter> or -<enter> to change scaling '
    colors = 30, 34, 35, 91, 93, 97
    chars: str = ' :%#\t#%:'
    gradient: list = []
    for bg, fg in zip(colors, colors[1:]):
        for char in chars:
            if char == '\t':
                bg, fg = fg, bg
            else:
                gradient.append('\x1b[{};{}m{}'.format(fg, bg + 10, char))

    # main loop
    for file in sound_files:
        # Calculate audio length
        length: float = WAVE(file).info.length / number_of_blocks

        # Plays audio on dediacted channel
        data, fs = read(file, dtype='float32')
        play(data, fs, device=chan_id)
        try:
            samplerate = query_devices(chan_id, 'input')[
                'default_samplerate']

            delta_f = (upper_bound - lower_bound) / (width_plage - 1)
            fftsize = ceil(samplerate / delta_f)
            low_bin = floor(lower_bound / delta_f)

            def callback(indata, frames, time, status):
                if status:
                    text = ' ' + str(status) + ' '
                    print('\x1b[34;40m', text.center(width_plage, '#'),
                          '\x1b[0m', sep='')
                if any(indata):
                    magnitude = abs(fft.rfft(indata[:, 0], n=fftsize))
                    magnitude *= gain / fftsize
                    line = (gradient[int(clip(x, 0, 1) * (len(gradient) - 1))]
                            for x in magnitude[low_bin:low_bin + width_plage])
                    print(*line, sep='', end='\x1b[0m\n')
                else:
                    print('no input')

            with InputStream(device=chan_id, channels=1, callback=callback,
                             blocksize=int(samplerate * length), samplerate=samplerate):
                while True:
                    response = input()
                    if response in ('', 'q', 'Q'):
                        break
                    for ch in response:
                        if ch == '+':
                            gain *= 2
                        elif ch == '-':
                            gain /= 2
                        else:
                            print('\x1b[31;40m', usage_line.center(width_plage, '#'),
                                  '\x1b[0m', sep='')
                            break
        except KeyboardInterrupt:
            exit('Interrupted by user')
        except Exception as e:
            exit(type(e).__name__ + ': ' + str(e))


def show_available_devices() -> None:
    print(query_devices())
