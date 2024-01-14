from ..mputils import consume_queue, write_list_to_queue
from sharedtypes import (
    AudioFrame,
    AudioFrameQueue,
    FollowerOutputQueue,
)
from ..eprint import eprint
import numpy as np
from GP_models.helper import stable_nlml


class Follower:
    def __init__(
            self,
            # output queue
            follower_output_queue: FollowerOutputQueue,
            # Performance and Score info
            audio_frames_queue: AudioFrameQueue,
            frame_length: int,
            sample_rate: int,
            score: list,
            cov_dict: dict,
            window: int,
            # GP model
            T: float,
            v: float,
            M: int,
            sigma_f: float,
            sigma_n: float,
    ):

        self.follower_output_queue = follower_output_queue
        self.audio_frames_queue = audio_frames_queue
        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.score = score
        self.cov_dict = cov_dict
        self.window = window
        self.T = T
        self.v = v
        self.M = M
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n

        self.frame_duration = self.frame_length/self.sample_rate
        self.frame_times = np.linspace(
            0, self.frame_duration, self.frame_length)

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        self.alignment()
        self.__log("Finished...")

    def alignment(self):
        """
        Performs score following
        Writes to self.follower_output_queue
        """
        # Check score
        if not self.score:
            raise ValueError(f"Score states are empty")

        # Begin score following — initiate values
        state_number, audio_frame_number = 0, 0
        self.follower_output_queue.put((state_number, audio_frame_number))

        frame = self.audio_frames_queue.get()
        if frame is None:
            self.follower_output_queue.put(None)
            return

        while True:
            # If we reach the end of the score
            if state_number == len(self.score):
                self.follower_output_queue.put(None)
                return

            # Get new audio frame
            frame = self.audio_frames_queue.get()
            if frame is None:
                self.follower_output_queue.put(None)
                return

            # If amplitude is great enough, perform alignment
            if np.sum(abs(frame)) > 10:  # TODO check this value then make it a default argument value
                probabilities = []
                num_lookahead = min(
                    len(self.score) - state_number, self.window)  # TODO check there isn't a plus one here
                for i in range(num_lookahead):
                    probabilities.append(stable_nlml(time_samples=self.frame_times, Y=frame,
                                         T=self.T, v=self.v, M=self.M, sigma_f=self.sigma_f, sigma_n=self.sigma_n, normalised=False, f=self.score[state_number+i], cov_dict=self.cov_dict))  # TODO make variables be parsed arguments
                priors = np.ones(num_lookahead)
                probabilities = np.array(probabilities)
                probabilities = probabilities * priors
                print(probabilities, flush=True)

                index = np.argmin(probabilities)

                state_number += index

            audio_frame_number += 1
            self.follower_output_queue.put((state_number, audio_frame_number))

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
