from tap import Tap
from sys import exit
from typing import Optional
import lib.constants as const
from os import path


class Arguments(Tap):

    perf_wave_path: Optional[str] = None  # Path to performance WAVE file.

    # Path to score MIDI.
    score_midi_path: Optional[str] = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/midi_files/Bach_1.midi'

    # Sample rate to synthesise score and load performance wave file.
    sample_rate: int = const.DEFAULT_SAMPLE_RATE
    frame_length: int = const.DEFAULT_FRAME_LENGTH

    # GP hyperparameters
    M: int = const.DEFAULT_M
    sigma_f: float = const.DEFAULT_SIGMA_F
    sigma_n: float = const.DEFAULT_SIGMA_N
    T: float = const.DEFAULT_T
    v: float = const.DEFAULT_V
