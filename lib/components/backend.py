from ..eprint import eprint
import sys
from lib.sharedtypes import (
    FollowerOutputQueue,
    MultiprocessingConnection,
)
import socket


class Backend:
    """
    Places tuples of (state_number, audio_frame_number) onto the follwer output queue
    """

    def __init__(
            self,
            follower_output_queue: FollowerOutputQueue,
            performance_stream_start_conn: MultiprocessingConnection,
            score_states: list,
            hop_length: int,
            frame_length: int,
            sample_rate: int,
            backend_output: str,
    ):
        self.follower_output_queue = follower_output_queue
        self.performance_stream_start_conn = performance_stream_start_conn
        self.hop_len = hop_length
        self.frame_len = frame_length
        self.score_states = score_states
        self.sample_rate = sample_rate
        self.backend_output = backend_output

        # For the case when using UDP to be used in conjunction with the Fliipy Qualitiative Testbench scorer
        if self.backend_output[:4] == "udp:":
            reduced_backend_output = self.backend_output[4:]
            # try to parse UDP IP and port
            address_port = reduced_backend_output.split(":")
            if len(address_port) != 2:
                raise ValueError(
                    f"Unknown `backend_output`: {self.backend_output}")
            self.addr = str(address_port[0])
            self.port = int(address_port[1])

            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.__log(
                f"Backend will be outputting via UDP to {self.addr}:{self.port}")
        elif self.backend_output == "stderr":
            pass
        else:
            raise ValueError(
                f"Unknown `backend_output`: {self.backend_output}")

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        self.__start_timestamp()
        self.__log("Finished")

    def __start_timestamp(self):
        prev_state = -1
        while True:
            path = self.follower_output_queue.get()
            if path is None:
                return

            state = path[0]
            score_time = path[1]

            # if state > prev_state:  # TODO atm we are assuming no backtracking
            if True:  # state > prev_state:
                # timestamp_s = self.__get_online_timestamp(state)
                # Output time! TODO this is where you change the code to make it print to a port!!!s
                # TODO make this back to eprint once done
                # print(
                #     f"state: {state}, audio frame: ?, timestamp: {score_time}", flush=True)
                if self.backend_output[:4] == "udp:":
                    self.__socket.sendto(
                        str(state).encode(), (self.addr, self.port))
                prev_state = state

    def __get_online_timestamp(self, state: int) -> float:
        return float(state * self.hop_len) / self.sample_rate

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
