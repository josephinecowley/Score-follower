from tap import Tap
from sys import exit
from typing import Optional
import lib.constants as const
from os import path
from sys import exit
from .eprint import eprint
from lib.sharedtypes import Mode


class Arguments(Tap):

    # Path to performance WAV file.
    # '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/fugue.wav'
    # '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/whole_bach_prelude.wav'
    perf_wave_path: Optional[str] = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/Bach_3.wav'

    # Path to score MIDI.
    # '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/midi_files/Bach_1.midi'
    score_midi_path: Optional[str] = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/midi_files/Bach_1.midi'
    # Sample rate to synthesise score and load performance wave file.
    sample_rate: int = const.DEFAULT_SAMPLE_RATE
    frame_length: int = const.DEFAULT_FRAME_LENGTH
    hop_length: int = const.DEFAULT_HOP_LENGTH
    max_duration: Optional[float] = 60

    # GP hyperparameters
    M: int = const.DEFAULT_M
    sigma_f: float = const.DEFAULT_SIGMA_F
    sigma_n: float = const.DEFAULT_SIGMA_N
    T: float = const.DEFAULT_T
    v: float = const.DEFAULT_V

    # Follower parameters
    window: int = 6
    back_track: int = 0
    mode: Mode = "viterbi"  # TODO to implement also "oltw" and "viterbi"
    scale_factor: float = 1
    # Whether to take into account state duration model transitions
    state_duration_model: bool = False
    # Either `stderr` or `udp:<HOSTNAME>:<PORT>` for UDP sockets + stderr
    backend_output: str = "udp:127.0.0.1:4000"
    max_run_count: int = 100
    threshold: float = 10.0
    sustain: bool = False

    # Miscellaneous
    # When streaming performance, reduce sleep time between streaming slices as sleeping is not entirely precise.
    sleep_compensation: float = 0.0025
    player_delay: float = 2.3
    # Whether to play the performance audio file when started. Requires `simulate_performance` to be set to True.
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
        #     else:
        #         self.score_midi_path = path.abspath(self.perf_wave_path)

        if self.score_midi_path:
            if not path.isfile(self.score_midi_path):
                self.__log_and_exit(
                    f"Score MIDI file ({self.score_midi_path}) does not exist"
                )
        #     else:
        #         self.score_midi_path = path.abspath(self.score_midi_path)
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
