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
        threshold: int,

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
        self.threshold: int = 75,

        self.cov_dict = cov_dict
        # TODO need to change the rest of the repo's naming convention so we dont muddle frames and samples
        self.time_samples = frame_times
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.T = T
        self.v = v
        self.M = M
        self.frame_length = frame_length

        # Viterbi data structures
        self.K = len(score)  # Total number of states in HMM

        # Initialise transmission matrix
        self.transmission = np.full((self.K, self.K), -np.inf)
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
        self.i = 0  # columns of gamma matrix

        self.__log("Initialised successfully")

    def follow(self):
        """
        Performs score following using an on-line implementation of the Viterbi algorithm.
        Writes to self.follower_outpu_queue
        """

        # Get the first audio frame and check not None
        frame = self.get_next_frame()
        if frame is None:
            self.follower_output_queue.put(None)
            return

        # Start at state 0
        lml = -helper.stable_nlml(self.time_samples, frame, M=self.M, normalised=False,
                                  f=self.score[0], T=self.T, v=self.v, cov_dict=self.cov_dict)
        self.gamma[0, 0] = lml  # Initialise probability of first audio sample

        while True:

            # Terminate if final state reached
            if self.max_s == len(self.score) - 1:
                self.follower_output_queue.put(None)
                return

            frame = self.get_next_frame()
            if frame is None:
                self.follower_output_queue.put(None)
                return

            k0_index = self.chunk * self.step
            for k in range(k0_index, k0_index + self.window):
                lml = -helper.stable_nlml(self.time_samples, frame, M=9, normalised=False,
                                          f=self.score[k], T=self.T, v=self.v, cov_dict=self.cov_dict)
                same_state = lml + \
                    self.gamma[k, self.i-1] + self.transmission[k, k]
                advance_state = lml + \
                    self.gamma[k-1, self.i-1] + self.transmission[k-1, k]

    def get_next_frame(self):
        """
        Check the frame is above threshold value, if not continue to get frames. 
        If frame is None, return None to output queue and terminate.
        """
        frame = self.audio_frames_queue.get()
        while np.sum(np.abs(frame)) < self.threshold:
            self.__log(f"Amplitude too small, moving onto next audio frame")
            frame = self.audio_frames_queue.get()

        return frame

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
