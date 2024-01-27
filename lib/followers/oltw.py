from ..eprint import eprint
import numpy as np
from GP_models.helper import stable_nlml
from sharedtypes import (
    AudioFrame,
    AudioFrameQueue,
    FollowerOutputQueue,
)
from enum import Enum
from typing import Set, Tuple
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
        max_run_count: int,

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
        self.max_run_count = max_run_count

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
            (len(self.score), self.frame_length), dtype=np.float64)
        self.R = np.ones((len(self.score), len(self.score)))

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
        self.__save_lml_to_R(i, j, lml)

        while True:
            # Exit program if last score state is reached
            if j == len(self.score) - 1:
                self.follower_output_queue.put(None)
                return

            # Get the next direction to increase the values of i or j (depending on the relative values of i, i_prime and j, j_prime)
            current = self.__get_next_drn(i, j, i_prime, j_prime, previous)

            # Now extend the raw lml cost matrix R

            # If need to incremenet i
            if Direction.I in current:
                # increment i
                i += 1
                # obtain the new audio frame
                p_i = self.audio_frames_queue.get()
                if p_i is None:
                    self.follower_output_queue.put(None)
                    return
                self.__save_p_i(i, p_i)

                # Compute the required R elements
                for J in range(max(0, j-self.window + 1), j + 1):
                    s_J = self.score[J]
                    lml = -helper.stable_nlml(self.frame_times, p_i, cov_dict=self.cov_dict, M=self.M,
                                              sigma_f=self.sigma_f, f=s_J, sigma_n=self.sigma_n, T=self.T, v=self.v, normalised=False)
                    self.__save_lml_to_R(i, J, lml)

            # If need to increment j
            if Direction.J in current:
                # Increment j
                j += 1
                # Obtain the new score state
                s_j = self.score[j]
                # Compute the required R elements
                for I in range(max(0, i - self.window + 1), i+1):
                    p_I = self.P[I]
                    lml = -helper.stable_nlml(self.frame_times, p_I, cov_dict=self.cov_dict, M=self.M,
                                              sigma_f=self.sigma_f, f=s_j, sigma_n=self.sigma_n, T=self.T, v=self.v, normalised=False)
                    self.__save_lml_to_R(I, j, lml)

            # Update run_count
            if current == previous and previous != DIR_IJ:
                self.run_count += 1
            else:
                self.run_count = 1
            # now set the current direction to the previous one, ready for the next iteration
            previous = current

            # Update the path values (i_prime and j_prime) and write to the output queue
            i_prime, j_prime = self.__get_path_values(i, j)

            # Finally, put path values on the follower output queue
            self.follower_output_queue.put((i_prime, j_prime))

    def __save_p_i(self, i: int, audio_frame: np.ndarray):
        """ Save audio frame to 2d array for later reference"""

        # If we have reached the end of the allocated space, append 50% more space
        if i >= self.P.shape[0]:
            length_to_append = int(0.5 * self.P.shape[0])
            self.P = np.append(self.P, np.zeros(
                (length_to_append, self.frame_length), dtype=np.float64), axis=0)

        # Else simply save the result into the 2d array
        self.P[i] = audio_frame

    def __save_lml_to_R(self, i, j, lml):
        """
        Save the lml value to the raw cost matrix R at R[i,j]
        """
        # If we have reached the end of the allocated space, append 50% more space
        if i >= self.R.shape[0]:
            # length to append in audio frames (i) direction
            length_to_append = int(0.5 * self.P.shape[0])
            self.R = np.append(self.R,
                               np.ones(length_to_append, len(self.score),
                                       dtype=np.float64) * np.inf,
                               axis=0)
        if (i, j) == (0, 0):
            self.R[i][j] = lml
        else:
            self.R[i][j] = lml + min(
                self.__R_get(i - 1, j - 1),
                self.__R_get(i-1, j),
                self.__R_get(i, j-1),
            )

    def __R_get(self, i: int, j: int) -> np.float64:
        """
        Return values of the lml Raw cost matrix for R[i, j]
        """
        # check we are access
        if i >= self.R.shape[0] or j >= self.R.shape[1]:
            raise ValueError(
                f"Values {i} or {j} are out of range for the matrix shape {self.R.shape}."
            )
        if i < 0 or j < 0:
            return np.float64(-np.inf)

        return self.R[i][j]

    def __get_next_drn(self, i: int, j: int, i_prime: int, j_prime: int, previous: Set[Direction]) -> Set[Direction]:
        """ Return the next direction to commence."""

        # If we are smaller than the window, we should keep increaseing in both directions
        if i < self.window:
            return DIR_IJ

        # If we have exceeded the max_run_count, then we should force it to progress in the opposite direction to avoid 'running away'
        elif self.run_count > self.max_run_count:
            if previous == DIR_I:
                return DIR_J
            else:
                return DIR_I

        if i_prime < i:
            return DIR_J
        elif j_prime < j:
            return DIR_I
        return DIR_IJ

    def __get_path_values(self, i: int, j: int) -> Tuple[int, int]:
        """Return a tuple of the path values (i,j)"""

        i_prime, j_prime = (i, j)
        max_R = -np.inf
        curr_i = i
        while curr_i >= 0 and curr_i > (i-self.window):
            if self.R[curr_i][j] > max_R:
                i_prime, jPrime = (curr_i, j)
                max_R = self.R[i_prime, j_prime]
            curr_i -= 1
        curr_j = j - 1
        while curr_j >= 0 and curr_j > (j - self.window):
            if self.R[i][curr_j] > max_R:
                i_prime, j_prime = (i, curr_j)
                max_R = self.R[i_prime][j_prime]
            curr_j -= 1
        return i_prime, j_prime

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
