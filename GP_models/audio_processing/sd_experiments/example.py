#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys
import threading

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
SAMPLE_SIZE = 1000
HOP_SIZE = 2


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])


parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-hop', '--hopsize', type=float, default=HOP_SIZE, metavar='HOP',
    help='number of samples between extraction(default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, default=SAMPLE_SIZE, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, default=SAMPLE_RATE, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)

q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    global count
    if status:
        print(status, file=sys.stderr)
    if count % args.hopsize == 0:
        q.put(indata[:], time)
    count += 1


def location():
    while True:
        try:
            data, time = q.get_nowait()
        except queue.Empty:
            break
        print(data)


try:

    stream = sd.InputStream(
        blocksize=args.blocksize,
        samplerate=args.samplerate, callback=audio_callback)
    # ani = FuncAnimation(fig, location, interval=args.interval, blit=True)
    with stream:
        location_thread = threading.Thread(target=location(), daemon=True)
        location_thread.start()

    # plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
