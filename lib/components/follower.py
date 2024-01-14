from ..mputils import consume_queue, write_list_to_queue
from sharedtypes import (
    AudioFrame,
    AudioFrameQueue,
    FollowerOutputQueue,
)
from ..eprint import eprint
import numpy as np


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
    ):

        self.follower_output_queue = follower_output_queue
        self.audio_frames_queue = audio_frames_queue
        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.score = score
        self.cov_dict = cov_dict
        self.window = window
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

        # Begin score following
        i = 0
        self.follower_output_queue.put((i, i))
        # Step 2
        p_i = self.audio_frames_queue.get()
        if p_i is None:
            self.follower_output_queue.put(None)
            return
        while True:
            if i == 100 - 1:
                self.follower_output_queue.put(None)
                return

            p_i = self.audio_frames_queue.get()
            if p_i is None:
                self.follower_output_queue.put(None)
                return

            i += 1
            self.follower_output_queue.put((i, i))

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
