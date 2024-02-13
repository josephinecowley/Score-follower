from ..eprint import eprint
import numpy as np
from GP_models.helper import stable_nlml
from lib.sharedtypes import (
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
        time_to_next: list,

        # Alignment parameters
        hop_length: int,
        window: int,
        threshold: int,
        state_duration_model: bool,

        # GP model hyperparameters
        sigma_f: float,
        sigma_n: float,
        cov_dict: dict,
        T: float,
        v: float,
        M: int,
        frame_length: int,
        frame_times: np.ndarray,

    ):

        self.follower_output_queue = follower_output_queue
        self.audio_frames_queue = audio_frames_queue
        self.score = score
        self.time_to_next = time_to_next

        self.hop_length = hop_length
        self.window = window
        self.threshold = threshold
        self.state_duration_model = state_duration_model

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

        # TODO— change this 1000 to be a number updated dynamically.With hop length 5000, ~500 would be one minute
        self.gamma = np.full((self.K, 1000), -np.inf,
                             'd')  # Probability matrix
        self.delta = np.zeros((self.K, 1000), 'B')  # Back pointers

        # # Initialise transmission matrix
        # self.transmission = np.full((self.K, self.K), -np.inf)
        # for i in range(self.K-1):
        #     self.transmission[i][i], self.transmission[i][i +
        #                                                   1] = np.log(0.5), np.log(0.5)
        # self.transmission[-1][-1] = np.log(1)
        # TODO need to deal with next transitiono for the final state in piece somehow, now that ive deleted the transition matrix

        # On-line viterbi parameters
        self.max_s = 0  # maximum likelihood current state
        self.chunk = 0
        self.step = window//3
        self.i = 0  # columns of gamma matrix

        # State duration model parameters
        # This keeps track of state durationm, d, the cumulative number of times we stay in state max_s
        self.counter = []

        # This keeps a running updated conversion rate (i.e. a kind of tempo measure)
        self.conversion_rate = self.hop_length / self.time_to_next[0]

        self.__log("Initialised successfully")

    def follow(self):
        """
        Performs score following using an on-line implementation of the Viterbi algorithm.
        Writes to self.follower_outpu_queue
        """
        d = 1  # We initialise state duration d to 1

        # Get the first audio frame and check not None
        frame = self.get_next_frame()
        if frame is None:
            self.follower_output_queue.put(None)
            return

        # Start at state 0
        lml = -helper.stable_nlml(self.time_samples, frame, M=self.M, normalised=False,
                                  f=self.score[0], T=self.T, v=self.v, cov_dict=self.cov_dict)
        lml_scaled = np.sign(lml) * np.abs(lml)**0.05
        # Initialise probability of first audio sample
        self.gamma[0, 0] = lml_scaled
        print("lml_scaled ", lml_scaled)

        advance_transition = np.log(0.5)
        self_transition = np.log(0.5)

        while True:

            # Terminate if final state reached
            if self.max_s == len(self.score) - 1:
                self.follower_output_queue.put(None)
                return

            # Get next audio frame and increment i
            frame = self.get_next_frame()
            self.i += 1
            if frame is None:
                self.follower_output_queue.put(None)
                return

            # Re-calculate transmission probabilities if taking into account state duration models
            if self.state_duration_model:
                expected = self.conversion_rate * \
                    self.time_to_next[self.max_s] / 1000
                p = 1 / expected  # probability
                q = 1 - p
                advance_transition = np.log(
                    np.sum([q**i * p for i in range(d)]))
                self_transition = np.log(1 - advance_transition)

            # Iterate through the states
            k0_index = self.chunk * self.step
            for k in range(k0_index, k0_index + self.window):
                lml = -helper.stable_nlml(self.time_samples, frame, M=9, normalised=False,
                                          f=self.score[k], T=self.T, v=self.v, cov_dict=self.cov_dict)
                lml_scaled = np.sign(lml) * np.abs(lml)**0.05

                print("lml scaled ", lml_scaled)
                same_state = lml_scaled + \
                    self.gamma[k, self.i-1] + self_transition
                advance_state = lml_scaled + \
                    self.gamma[k-1, self.i-1] + advance_transition
                self.gamma[k, self.i] = np.max([same_state, advance_state])

            # Determine most likely state
            new_s = np.argmax(self.gamma[:, self.i])

            # If required, update state duration d
            if self.state_duration_model:
                if new_s == self.max_s:
                    d += 1  # If still in same state, keep d the same
                else:
                    self.counter.append(d)
                    conversion_rates = 1000 * np.array(self.counter) / np.array(
                        self.time_to_next[:len(self.counter)])  # Multiply by 1000 to make seconds
                    # We take the running mean average
                    self.conversion_rate = np.mean(conversion_rates)
                    d = 1

            self.max_s = new_s

            # Print to outut queue
            print(self.max_s, self.i, flush=True)
            self.follower_output_queue.put((self.max_s, self.i))

            # Update chunk
            if self.max_s >= k0_index + self.window - self.step:
                self.chunk += 1

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
