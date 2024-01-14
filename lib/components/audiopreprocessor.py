from ..mputils import write_list_to_queue

from ..eprint import eprint
from sharedtypes import ExtractedFeature, ExtractedFeatureQueue, ModeType
from typing import Callable, Optional, Dict, List, Union

import multiprocessing as mp
import librosa  # type: ignore
import time
import numpy as np


class Slicer:
    def __init__(
        self,
        wave_path: str,
        hop_length: int,
        frame_length: int,
        sample_rate: int,
        slice_queue: ExtractedFeatureQueue,
        sleep_compensation: float,
    ):
        self.wave_path = wave_path
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.slice_queue = slice_queue
        self.sleep_compensation = sleep_compensation

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        audio_stream = librosa.stream(
            path=self.wave_path,
            block_length=1,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            mono=True,
            fill_value=0,
        )

        # before starting, sleep for frame_length
        self.__sleep(
            self.frame_length, time.perf_counter() + 0.2)

        for s in audio_stream:
            pre_sleep_time = time.perf_counter()
            self.slice_queue.put(s)
            # sleep for hop length
            self.__sleep(self.hop_length, pre_sleep_time)

        self.slice_queue.put(None)  # end
        self.__log("Finished")

    def __sleep(self, samples: int, pre_sleep_time: float):
        sleep_time = float(samples) / self.sample_rate
        time.sleep(
            sleep_time
            - (time.perf_counter() - pre_sleep_time)
            - self.sleep_compensation
        )

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")


class FeatureExtractor:
    def __init__(
            self,
            slice_queue: ExtractedFeatureQueue,
            output_queue: ExtractedFeatureQueue,
            hop_length: int,
            frame_length: int,
            sample_rate: int,
    ):
        self.slice_queue = slice_queue
        self.output_queue = output_queue
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.sample_rate = sample_rate

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        while True:
            sl: Optional[np.ndarray] = self.slice_queue.get()
            if sl is None:
                break
            self.output_queue.put(sl)
        self.output_queue.put(None)  # end
        self.__log("Finished")

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")


# TODO continue!@!!


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        frame_length: int,
        # slicer
        wave_path: str,
        sleep_compensation: float,
        output_queue: ExtractedFeatureQueue,

    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.wave_path = wave_path
        self.sleep_compensation = sleep_compensation
        self.output_queue = output_queue

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        slice_queue: ExtractedFeatureQueue = mp.Queue()

        online_slicer_proc: Optional[mp.Process] = None

        slicer = Slicer(
            wave_path=self.wave_path,
            hop_length=self.hop_length,
            frame_length=self.frame_length,
            sample_rate=self.sample_rate,
            slice_queue=slice_queue,
            sleep_compensation=self.sleep_compensation,
        )

        online_slicer_proc = mp.Process(target=slicer.start)
        online_slicer_proc.start()

        feature_extractor = FeatureExtractor(
            slice_queue=slice_queue,
            output_queue=self.output_queue,
            hop_length=self.hop_length,
            frame_length=self.frame_length,
            sample_rate=self.sample_rate,
        )
        feature_extractor_proc = mp.Process(target=feature_extractor.start)
        feature_extractor_proc.start()

        online_slicer_proc.join()
        feature_extractor_proc.join()
        self.__log("Finished")

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
