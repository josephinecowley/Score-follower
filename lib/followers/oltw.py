from ..eprint import eprint
import numpy as np
from GP_models.helper import stable_nlml
from sharedtypes import (
    AudioFrame,
    AudioFrameQueue,
    FollowerOutputQueue,
)
from enum import Enum
from typing import Set
from GP_models import helper


class Direction(Enum):
    I = 1
    J = 2


DIR_I = set([Direction.I])
DIR_J = set([Direction.J])
DIR_IJ = set([Direction.I, Direction.J])


class Oltw:
    def __init__(
        self,

        # Score and performance parameters
        follower_output_queue: FollowerOutputQueue,
        audio_frames_queue: AudioFrameQueue,
        score: list,

        # Score following parameters
        back_track: int,
        # TODO maybe get rid of this and instead declare locally--depends on what makes sense
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

        self.back_track = back_track
        self.window = window

        self.cov_dict = cov_dict
        self.frame_times = frame_times
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.T = T
        self.v = v
        self.M = M
        self.frame_length = frame_length

        # A 2d array which saves all the audio frames, initiated to the length of the score
        self.P = np.zeros(
            (len(self.score), len(self.frame_length)), dtype=np.float64)

        self.__log("Initialised successfully")

    def follow(self):
        """
        Performs score following using a oltw and raw lml values
        Writes to self.follower_output_queue
        """

        # First initiate values
        i, j = 0, 0
        current: Set[Direction] = set()
        previous: Set[Direction] = set()
        i_prime, j_prime = 0, 0

        # Output the initial values to the follower output queue
        self.follower_output_queue.put((i_prime, j_prime))

        # Get the first audio frame and save it to a matrix
        p_i = self.audio_frames_queue.get()
        if p_i is None:
            self.follower_output_queue.put(None)
            return
        self.__save_p_i(i, p_i)

        # Get the first score state
        s_j = self.score[j]

        # Now calcualte raw likelihoods to populate R
        lml = -helper.stable_nlml(self.frame_times, p_i, cov_dict=self.cov_dict, M=self.M,
                                  sigma_f=self.sigma_f, f=s_j, sigma_n=self.sigma_n, T=self.T, v=self.v, normalised=False)
        self.__save_lml_to_R(lml)

    def __save_p_i(self, i: int, audio_frame: np.ndarray):
        """ Save audio frame to 2d array for later reference"""

        # If we have reached the end of the allocated space, append 50% more space
        if i >= self.P.shape[0]:
            length_to_append = int(0.5 * self.P.shape[0])
            self.P = np.append(self.P, np.zeros(
                (length_to_append, self.frame_length), dtype=np.float64), axis=0)

        # Else simply save the result into the 2d array
        self.P[i] = audio_frame

    def __save_lml_to_R(self, lml):
        pass
