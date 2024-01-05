#!/usr/bin/env python3
"""Plot the LML

"""
import argparse
import queue
import sys
import time

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

# Constants
SAMPLE_RATE = 44100.0

# Parse arguments blocksize and samplerate
parser = argparse.ArgumentParser(add_help=False)
args, remaining = parser.parse_known_args()
parser.add_argument(
    '-b', '--blocksize', type=int, default=2000, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
args = parser.parse_args(remaining)

total_audio = []

times = list()
print(type(times))


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    global times
    if status:
        print(status, file=sys.stderr)

    print(np.sum(abs(indata)), indata)


try:

    # Start the keyboard listener in a separate thread

    if args.samplerate is None:
        args.samplerate = SAMPLE_RATE
    callback_interval = 0.5  # Callback every half second

    stream = sd.InputStream(
        channels=1, samplerate=args.samplerate, callback=audio_callback, blocksize=2000)

    with stream:
        while True:
            # Sleep for a short duration to release the GIL and allow the callback to run
            # sd.sleep(100)
            sd.sleep(1*1000)

except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
