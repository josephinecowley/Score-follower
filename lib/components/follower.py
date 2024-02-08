from ..mputils import consume_queue, write_list_to_queue
from lib.sharedtypes import (
    AudioFrame,
    AudioFrameQueue,
    FollowerOutputQueue,
    Mode,
)
from ..eprint import eprint
import numpy as np
from lib.followers.basic import Basic
from lib.followers.oltw import Oltw
from lib.followers.viterbi import Viterbi


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
            time_to_next: list,
            cov_dict: dict,
            window: int,
            back_track: int,
            mode: Mode,
            max_run_count: int,
            threshold: float,

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
        self.time_to_next = time_to_next
        self.cov_dict = cov_dict
        self.window = window
        self.back_track = back_track
        self.T = T
        self.v = v
        self.M = M
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.mode = mode
        self.max_run_count = max_run_count
        self.threshold = threshold

        self.frame_duration = self.frame_length/self.sample_rate
        self.frame_times = np.linspace(
            0, self.frame_duration, self.frame_length)

        follower_options: dict = {
            "basic": self.start_basic,
            "oltw": self.start_oltw,
            "viterbi": self.start_viterbi,
        }
        self.__start = follower_options.get(self.mode)
        if self.__start is None:
            raise ValueError(
                f"Invalid score following mode {self.mode}."
            )

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        self.__start()
        self.__log("Finished...")

    def start_viterbi(self):
        viterbi_follower = Viterbi(
            follower_output_queue=self.follower_output_queue,
            audio_frames_queue=self.audio_frames_queue,
            score=self.score,
            time_to_next=self.time_to_next,
            frame_times=self.frame_times,
            window=self.window,
            threshold=self.threshold,
            sigma_f=self.sigma_f,
            sigma_n=self.sigma_n,
            cov_dict=self.cov_dict,
            T=self.T,
            v=self.v,
            M=self.M,
            frame_length=self.frame_length,
        )
        viterbi_follower.follow()

    def start_basic(self):  # TODO rename basic to greedy
        basic_follower = Basic(
            follower_output_queue=self.follower_output_queue,
            audio_frames_queue=self.audio_frames_queue,
            score=self.score,
            sigma_f=self.sigma_f,
            sigma_n=self.sigma_n,
            cov_dict=self.cov_dict,
            back_track=self.back_track,
            frame_times=self.frame_times,
            window=self.window,
            T=self.T,
            v=self.v,
            M=self.M,
        )
        basic_follower.follow()

    def start_oltw(self):
        oltw_follower = Oltw(
            follower_output_queue=self.follower_output_queue,
            audio_frames_queue=self.audio_frames_queue,
            score=self.score,
            sigma_f=self.sigma_f,
            sigma_n=self.sigma_n,
            cov_dict=self.cov_dict,
            back_track=self.back_track,
            frame_times=self.frame_times,
            window=self.window,
            T=self.T,
            v=self.v,
            M=self.M,
            max_run_count=self.max_run_count,
            frame_length=self.frame_length,
        )
        oltw_follower.follow()

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
