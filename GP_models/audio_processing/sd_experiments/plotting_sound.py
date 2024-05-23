#!/usr/bin/env python3
"""Plot the LML

"""
import argparse
import queue
import sys
import time
import numpy as np

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import librosa
from ...helper import plot_audio
from .eprint import eprint


class Preprocessor:
    def __init__(
        self,
        block_size: int,
        samplerate: int

    ):
        self.block_size = block_size
        self.samplerate = samplerate
        self.audio_duration = self.block_size/self.samplerate
        self.times = np.linspace(0, self.audio_duration, self.block_size)

    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        global y
        if status:
            print(status, file=sys.stderr)
        q.put(indata)

    def update_plot(self):
        """This is called by matplotlib for each plot update.

        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.

        """
        global plotdata
        while True:
            try:
                data = q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            plotdata = np.roll(plotdata, -shift, axis=0)
            plotdata[-shift:, :] = data
        for column, line in enumerate(lines):
            line.set_ydata(plotdata[:, column])
        return lines

    def onset_detect(self, data):
        data = np.array(data).flatten()
        sample_indices = librosa.onset.onset_detect(
            y=data, post_avg=5, wait=1,  sr=self.samplerate, delta=0.5, units='samples')
        print(sample_indices)

    def start(self):
        try:
            # Parse arguments blocksize and samplerate
            # parser = argparse.ArgumentParser(add_help=False)
            # args, remaining = parser.parse_known_args()
            # parser.add_argument(
            #     '-b', '--blocksize', type=int, default=self.block_size, help='block size (in samples)')
            # parser.add_argument(
            #     '-r', '--samplerate', type=float, help='sampling rate of audio device')
            # args = parser.parse_args(remaining)

            q = queue.Queue()

            fig, ax = plt.subplots()
            lines = ax.plot(self.times)
            ax.axis((0, len(plotdata), -1, 1))
            ax.set_yticks([0])
            ax.yaxis.grid(True)
            ax.tick_params(bottom='off', top='off', labelbottom='off',
                           right='off', left='off', labelleft='off')
            fig.tight_layout(pad=0)
            stream = sd.InputStream(
                channels=1, samplerate=self.samplerate, callback=self.audio_callback, blocksize=self.block_size)
            ani = FuncAnimation(fig, self.update_plot, interval=30, blit=True)
            with stream:
                while True:
                    plt.show()

        except Exception as e:
            parser.exit(type(e).__name__ + ': ' + str(e))
