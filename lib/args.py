from tap import Tap
from sys import exit
from typing import Optional
import lib.constants as const
from os import path
from sys import exit
from .eprint import eprint
from lib.sharedtypes import Mode


class Arguments(Tap):

    # Score follower
    perf_wave_path: Optional[str] = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/final/music_data/wav_files/bach_hymn.wav'
    score_midi_path: Optional[str] = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/final/music_data/midi_files/bach_hymn.midi'
    sample_rate: int = const.DEFAULT_SAMPLE_RATE
    frame_length: int = const.DEFAULT_FRAME_LENGTH
    hop_length: int = const.DEFAULT_HOP_LENGTH
    max_duration: Optional[float] = 300  # Time in seconds before timing out

    # GP hyperparameters
    M: int = const.DEFAULT_M
    sigma_f: float = const.DEFAULT_SIGMA_F
    sigma_n: float = const.DEFAULT_SIGMA_N
    T: float = const.DEFAULT_T
    v: float = const.DEFAULT_V

    # Follower parameters
    window: int = 6
    back_track: int = 0
    mode: Mode = "viterbi"  # Legacy: could add "oltw" and "viterbi" as well
    # Exponent to make likelihoods and tranition probabilities comparable
    scale_factor: float = 0.25
    state_duration_model: bool = False
    # Either `stderr` or `udp:<HOSTNAME>:<PORT>` for UDP sockets + stderr
    backend_output: str = "udp:127.0.0.1:4000"
    max_run_count: int = 100
    threshold: float = 800000000  # Mode 1: 800000000 Mode 2: 1
    sustain: bool = False

    sleep_compensation: float = 0.0025
    player_delay: float = 1.55  # The more laggy the computer is, use a larger this value
    play_performance_audio: bool = True

    def __log_and_exit(self, msg: str):
        self.__log(f"Argument Error: {msg}.")
        self.__log("Use the `--help` flag to show the help message.")
        exit(1)

    def sanitise(self):
        if self.perf_wave_path:
            if not path.isfile(self.perf_wave_path):
                self.__log_and_exit(
                    f"Performance wave file ({self.perf_wave_path}) does not exist"
                )

        if self.score_midi_path:
            if not path.isfile(self.score_midi_path):
                self.__log_and_exit(
                    f"Score MIDI file ({self.score_midi_path}) does not exist"
                )

        if self.sample_rate < 0:
            self.__log_and_exit(f"sample_length must be positive")

        if self.hop_length < 0:
            self.__log_and_exit(f"hop_length must be positive")

        if self.frame_length < 0:
            self.__log_and_exit(f"frame_length must be positive")

        if self.M < 0:
            self.__log_and_exit(f"M must be positive")

        if self.sigma_f < 0:
            self.__log_and_exit(f"sigma_f must be positive")

        if self.sigma_n < 0:
            self.__log_and_exit(f"sigma_n must be positive")

        if self.T < 0:
            self.__log_and_exit(f"T must be positive")

        if self.v < 0:
            self.__log_and_exit(f"v must be positive")

        if self.window < 0:
            self.__log_and_exit(f"window must be positive")

        if self.sleep_compensation < 0:
            self.__log_and_exit(f"sleep_compensation must be positive")

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
