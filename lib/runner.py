import numpy as np
from .args import Arguments

from .components.follower import Follower
from .components.backend import Backend
from collections import defaultdict
from .eprint import eprint
from midi.midi import process_midi_to_note_info, notes_to_chords, dict_to_frequency_list
from GP_models.helper import SM_kernel
from sharedtypes import (
    List,
    ExtractedFeature,
    ExtractedFeatureQueue,
    FollowerOutputQueue,
    MultiprocessingConnection,
    NoteInfo,
)
import multiprocessing as mp
from .components.audiopreprocessor import AudioPreprocessor
import time


class Runner:
    def __init__(self, args: Arguments):
        """
        Precondition: assuming args.sanitize() was called.
        """
        self.args = args
        self.frame_duration = self.args.frame_length/self.args.sample_rate
        self.frame_times = np.linspace(
            0, self.frame_duration, self.args.frame_length)
        self.__log(f"Initiated with arguments:\n{args}")

    def start(self):
        self.__log(f"STARTING")

        follower_output_queue: FollowerOutputQueue = mp.Queue()
        (
            parent_performance_stream_start_conn,
            child_performance_stream_start_conn,
        ) = mp.Pipe()
        P_queue: ExtractedFeatureQueue = mp.Queue()

        self.__log(f"Begin: preprocess score")
        score = self.__preprocess_score()
        self.__log(f"End: preprocess score")

        self.__log(f"Begin: precalculate covariance matrices")
        cov_dict = self.__precalculate_cov(score[:20])
        self.__log(f"End: precalculate covariance matrices")

        self.__log(f"Begin: initialise performance processor")
        perf_ap = self.__init_performance_processor(P_queue)
        self.__log(f"End: initialise performance processor")

        self.__log(f"Begin: initialise follower")
        follower = self.__init_follower(
            follower_output_queue, P_queue, score, cov_dict)
        self.__log(f"End: initialise follower")

        self.__log(f"Begin: initialise backend")
        backend = self.__init_backend(
            follower_output_queue,
            parent_performance_stream_start_conn,
            score,
        )
        self.__log(f"End: initialise backend")

        perf_ap_proc = mp.Process(target=perf_ap.start)
        follower_proc = mp.Process(target=follower.start)
        backend_proc = mp.Process(target=backend.start)

        # start from the back
        self.__log(f"Starting: backend")
        backend_proc.start()
        self.__log(f"Starting: follower")
        follower_proc.start()

        perf_start_time = time.perf_counter()

        self.__log(f"Starting: performance at {perf_start_time}")
        child_performance_stream_start_conn.send(perf_start_time)
        perf_ap_proc.start()

        backend_proc.join()
        self.__log("Joined: backend")
        follower_proc.join()
        self.__log("Joined: follower")
        perf_ap_proc.terminate()  # use terminate as sometimes it hangs forever
        self.__log("Joined: performance")

    def __init_performance_processor(
        self, P_queue: ExtractedFeatureQueue
    ) -> AudioPreprocessor:
        args = self.args
        ap = AudioPreprocessor(
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            frame_length=args.frame_length,
            wave_path=args.perf_wave_path,
            output_queue=P_queue,
            sleep_compensation=args.sleep_compensation,
        )
        return ap

    def __preprocess_score(self) -> list:
        """
        Return list of states
        """
        args = self.args
        note_info = process_midi_to_note_info(args.score_midi_path)
        self.__log("Finished getting note info from score midi")

        dic = notes_to_chords(note_info)
        self.__log("Finished getting chords from note info")

        score = dict_to_frequency_list(dic)
        self.__log("Finished getting states from chords")

        return score

    def __precalculate_cov(self, score: list) -> dict:
        """
        Return dictionary of covariance functions (wth noise!) to speed up score following
        """
        args = self.args
        cov_dict = {}

        for state in score:
            if str(state) not in cov_dict:
                cov_dict[str(state)] = SM_kernel(
                    self.frame_times, self.frame_times, M=args.M,  f=state, sigma_f=args.sigma_f, T=args.T, v=args.v) + args.sigma_n**2 * np.eye(args.frame_length)

        self.__log("Finished getting dictionary of covariance matrices")
        return cov_dict

    def __init_follower(
        self,
        follower_output_queue: FollowerOutputQueue,
        P_queue: ExtractedFeatureQueue,
        score: list,
        cov_dict: dict,
    ) -> Follower:
        args = self.args
        return Follower(
            follower_output_queue=follower_output_queue,
            P_queue=P_queue,
            score=score,
            cov_dict=cov_dict,
            window=args.window
        )

    def __init_backend(
        self,
        follower_output_queue: FollowerOutputQueue,
        performance_stream_start_conn: MultiprocessingConnection,
        score_states: list,
    ) -> Backend:
        args = self.args

        return Backend(
            follower_output_queue=follower_output_queue,
            performance_stream_start_conn=performance_stream_start_conn,
            score_states=score_states,
            hop_length=args.hop_length,
            frame_length=args.frame_length,
            sample_rate=args.sample_rate,
        )

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
