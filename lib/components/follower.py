from ..mputils import consume_queue, write_list_to_queue
from sharedtypes import (
    ExtractedFeature,
    ExtractedFeatureQueue,
    FollowerOutputQueue,
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

    def start(self):
        self.__log("Starting...")
        self.alignment()
        self.__log("Finished...")

    def alignment(self):
        """
        Performs score following
        Writes to self.follower_output_queue
        """
        # Perform type checking
        if not self.cov_dict:
            raise ValueError(f"Empty covariance matrix")
        if not self.score:
            raise ValueError(f"Score states are empty")
        i = 0
        self.follower_output_queue.put((i, i))
        # Step 2
        p_i = self.P_queue.get()
        if p_i is None:
            self.follower_output_queue.put(None)
            return
        while True:
            if i == 100 - 1:
                self.follower_output_queue.put(None)
                return

            p_i = self.P_queue.get()
            if p_i is None:
                self.follower_output_queue.put(None)
                return

            i += 1
            self.follower_output_queue.put((i, i))

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
