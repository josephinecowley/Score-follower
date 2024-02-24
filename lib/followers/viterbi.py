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
import time


class Viterbi:
    def __init__(
        self,

        # Score and performance parameters
        follower_output_queue: FollowerOutputQueue,
        audio_frames_queue: AudioFrameQueue,
        score: list,
        time_to_next: list,
        score_times: np.ndarray,

        # Alignment parameters
        hop_length: int,
        window: int,
        threshold: int,
        state_duration_model: bool,
        scale_factor: float,

        # GP model hyperparameters
        sigma_f: float,
        sigma_n: float,
        cov_dict: dict,
        T: float,
        v: float,
        M: int,
        frame_length: int,
        frame_times: np.ndarray,
        sample_rate: float,

    ):

        self.follower_output_queue = follower_output_queue
        self.audio_frames_queue = audio_frames_queue
        self.score = score
        self.time_to_next = time_to_next
        self.score_times = score_times

        self.hop_length = hop_length
        self.window = window
        self.threshold = threshold
        self.state_duration_model = state_duration_model
        self.scale_factor = scale_factor

        self.cov_dict = cov_dict
        # TODO need to change the rest of the repo's naming convention so we dont muddle frames and samples
        self.time_samples = frame_times
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.T = T
        self.v = v
        self.M = M
        self.frame_length = frame_length
        self.sample_rate = sample_rate

        self.__log("Initialised successfully")

    def follow(self):
        """
        Performs score following using an on-line implementation of the Viterbi algorithm.
        Writes to self.follower_outpu_queue
        """

        # Initialise on-line viterbi variables
        gamma = np.full((len(self.score), 1000), -np.inf, 'd')
        max_s = 0
        i = 0  # column number of gamma matrix
        chunk = 0  # keeps track of on-line viterbi window location
        step = self.window//3  # Threshold to trigger next chunk

        # Initialise state duration variables
        conversion_rate = self.sample_rate / self.hop_length

        counter = []  # This keeps track of state durations, d
        d = 1

        # Get the first audio frame and check not None
        frame = self.get_next_frame()
        if frame is None:
            self.follower_output_queue.put(None)
            return

        # Start at state 0
        lml = -helper.stable_nlml(self.time_samples, frame, M=self.M, normalised=False,
                                  f=self.score[0], T=self.T, v=self.v, cov_dict=self.cov_dict)
        lml_scaled = np.sign(lml) * np.abs(lml)**self.scale_factor
        gamma[0, 0] = lml_scaled

        advance_transition = np.log(0.5)
        self_transition = np.log(0.5)

        while True:

            # Terminate if final state reached
            if max_s == len(self.score) - 1:
                self.follower_output_queue.put(None)
                return

            # Get next audio frame and increment i
            frame = self.get_next_frame()
            i += 1
            if frame is None:
                self.follower_output_queue.put(None)
                return

            # Lengthen gamma matrix if needed
            if i > gamma.shape[1]:
                desired_len = int(i * 1.5)
                columns_to_add = desired_len - gamma.shape[1]
                gamma = np.append(gamma, np.full(
                    (len(self.score), columns_to_add), -np.inf, 'd'))

            # Re-calculate transmission probabilities if taking into account state duration models
            if self.state_duration_model:
                expected = conversion_rate * \
                    self.time_to_next[max_s] / 1000
                p = 1 / expected  # probability
                q = 1 - p
                advance_transition = np.sum([q**z * p for z in range(d)])
                self_transition = np.log(1 - advance_transition)
                advance_transition = np.log(advance_transition)
                print("self ", self_transition, "advance ", advance_transition)

            # Iterate through states in window
            k0_index = chunk * step
            for k in range(k0_index, k0_index + self.window):
                lml = -helper.stable_nlml(self.time_samples, frame, M=self.M, normalised=False,
                                          f=self.score[k], T=self.T, v=self.v, cov_dict=self.cov_dict)
                lml_scaled = np.sign(lml) * np.abs(lml)**self.scale_factor

                same_state = lml_scaled + \
                    gamma[k, i-1] + self_transition
                advance_state = lml_scaled + \
                    gamma[k-1, i-1] + advance_transition
                gamma[k, i] = np.max(
                    [same_state, advance_state])

            # Determine most likely state
            new_s = np.argmax(gamma[:, i])

            # If required, update state duration d
            if self.state_duration_model:
                if new_s == max_s:
                    d += 1  # If still in same state, keep d the same
                else:
                    counter.append(d)
                    conversion_rates = 1000 * np.array(counter) / np.array(
                        self.time_to_next[:len(counter)])
                    # We take the running mean average
                    conversion_rate = np.mean(conversion_rates)
                    d = 1

            max_s = new_s

            # Print to outut queue
            print(max_s, flush=True)
            self.follower_output_queue.put(
                (max_s, self.score_times[max_s]))  # also returning score times for legacy reasons TODO: delete at end of project

            # Update chunk
            if max_s >= k0_index + self.window - step:
                chunk += 1

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
