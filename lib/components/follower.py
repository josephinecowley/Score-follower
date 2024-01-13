from ..mputils import consume_queue, write_list_to_queue
from sharedtypes import (
    ExtractedFeature,
    ExtractedFeatureQueue,
    FollowerOutputQueue,
    ModeType,
)
from ..eprint import eprint
from typing import Callable, Dict, List


class Follower:
    def __init__(
            self,
            # output queue
            follower_output_queue: FollowerOutputQueue,
            # Performance and Score info
            P_queue: ExtractedFeatureQueue,
            score: list,
            cov_dict: dict,
            window: int,
    ):

        self.follower_output_queue = follower_output_queue
        self.P_queue = P_queue
        self.score = score
        self.cov_dict = cov_dict
        self.window = window

        self.__log("Initialised successfully")

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
