from tap import Tap
from sys import exit
from typing import Optional
from .constants import DEFAULT_SAMPLE_RATE, DEFAULT_FRAME_LENGTH
from os import path


class Arguments(Tap):

    perf_wave_path: Optional[str] = None  # Path to performance WAVE file.

    # Path to score MIDI.
    score_midi_path: Optional[str] = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/midi_files/Bach_1.midi'

    # Sample rate to synthesise score and load performance wave file.
    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_length: int = DEFAULT_FRAME_LENGTH
