from ..eprint import eprint
from typing import Callable, Iterator, List, Dict, Set
from sharedtypes import (
    FollowerOutputQueue,
    ModeType,
    NoteInfo,
    MultiprocessingConnection,
)
import time
import socket


class Backend:
    """
    Outputs results of alignment
    """

    def __init__(
            self,
            follower_output_queue: FollowerOutputQueue,
            performance_stream_start_conn: MultiprocessingConnection,
            score_states: list,
            hop_length: int,
            frame_length: int,
            sample_rate: int,
    ):
        self.follower_output_queue = follower_output_queue
        self.performance_stream_start_conn = performance_stream_start_conn
        self.hop_len = hop_length
        self.frame_len = frame_length
        self.score_states = score_states
        self.sample_rate = sample_rate

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        self.__start_timestamp()
        self.__log("Finished")

    def __start_timestamp(self):
        prev_s = -1
        while True:
            e = self.follower_output_queue.get()
            if e is None:
                return
            s = e[1]
            if (self.backend_backtrack and s != prev_s) or (
                not self.backend_backtrack and s > prev_s
            ):
                timestamp_s = self.__get_online_timestamp(s)
                # Output time! TODO this is where you change the code to make it print to a port!!!
                print(timestamp_s, flush=True)
                prev_s = s

    def __get_online_timestamp(self, s: int) -> float:
        return float(self.frame_len + (s - 1) * self.hop_len) / self.sample_rate

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
