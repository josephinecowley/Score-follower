from ..eprint import eprint
import numpy as np
from GP_models.helper import stable_nlml
from midi.sharedtypes import (
    AudioFrame,
    AudioFrameQueue,
    FollowerOutputQueue,
)
from enum import Enum
from typing import Set, Tuple
from GP_models import helper


class Viterbi:
    def __init__(
        self,

        # Score and performance parameters
        follower_output_queue: FollowerOutputQueue,
        audio_frames_queue: AudioFrameQueue,
        score: list,

        frame_times: np.ndarray,
        window: int,

        # GP model hyperparameters
        sigma_f: float,
        sigma_n: float,
        cov_dict: dict,
        T: float,
        v: float,
        M: int,
        frame_length: int,

    ):

        self.follower_output_queue = follower_output_queue
        self.audio_frames_queue = audio_frames_queue
        self.score = score

        self.window = window
        self.run_count: int = 1

        self.cov_dict = cov_dict
        # TODO need to change the rest of the repo's naming convention so we dont muddle frames and samples
        self.sample_times = frame_times
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.T = T
        self.v = v
        self.M = M
        self.frame_length = frame_length

        # Viterbi data structures
        self.K = len(score)  # Total number of states in HMM

        # Initialise transmission matrix
        self.T = np.full((self.K, self.K), -np.inf)
        for i in range(self.K-1):
            T[i][i], T[i][i+1] = np.log(0.5), np.log(0.5)
        T[-1][-1] = np.log(1)

        # TODO— change this 1000 to be a number updated dynamically.With hop length 5000, ~500 would be one minute
        self.gamma = np.full((self.K, 1000), -np.inf,
                             'd')  # Probability matrix
        self.delta = np.zeros((self.K, 1000), 'B')  # Back pointers

        self.max_s = 0  # maximum likelihood current state
        self.chunk = 0
        self.step = window//3

        self.__log("Initialised successfully")

    def follow(self):
        """
        Performs score following using an on-line implementation of the Viterbi algorithm.
        Writes to self.follower_outpu_queue
        """

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
