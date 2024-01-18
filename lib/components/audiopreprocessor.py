from ..mputils import write_list_to_queue

from ..eprint import eprint
from sharedtypes import AudioFrame, AudioFrameQueue
from typing import Callable, Optional, Dict, List, Union
import sounddevice as sd
import multiprocessing as mp
import librosa  # type: ignore
import time
import numpy as np


class Slicer:
    def __init__(
        self,
        wave_path: Optional[str],
        hop_length: int,
        frame_length: int,
        sample_rate: int,
        audio_frames_queue: AudioFrameQueue,
        sleep_compensation: float,
    ):
        self.wave_path = wave_path
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.audio_frames_queue = audio_frames_queue
        self.sleep_compensation = sleep_compensation

        self.__log("Initialised successfully")

    def start(self):
        if self.wave_path:  # If htere is a wave path, we will not do live return
            self.__log("Starting using performance recording...")
            audio_stream = librosa.stream(
                path=self.wave_path,
                block_length=1,
                # TODO hop length parameter didn't seem to be working so this is a current fix
                frame_length=self.hop_length,
                hop_length=self.hop_length,
                mono=True,
                fill_value=0,
            )

            # before starting, sleep for frame_length
            self.__sleep(
                self.frame_length, time.perf_counter() + 0.2)

            for audio_frame in audio_stream:
                pre_sleep_time = time.perf_counter()
                self.audio_frames_queue.put(audio_frame[:self.frame_length])
                # sleep for hop length
                self.__sleep(self.hop_length, pre_sleep_time)

            self.audio_frames_queue.put(None)  # end
            self.__log("Finished")

        else:
            self.__log("Starting to listen...")
            self.audio_frames_queue.put(None)  # end
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


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        frame_length: int,
        # slicer
        wave_path: Optional[str],
        sleep_compensation: float,
        audio_frames_queue: AudioFrameQueue,

    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.wave_path = wave_path
        self.sleep_compensation = sleep_compensation
        self.audio_frames_queue = audio_frames_queue

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")

        slicer = Slicer(
            wave_path=self.wave_path,
            hop_length=self.hop_length,
            frame_length=self.frame_length,
            sample_rate=self.sample_rate,
            audio_frames_queue=self.audio_frames_queue,
            sleep_compensation=self.sleep_compensation,
        )

        online_slicer_proc = mp.Process(target=slicer.start)
        online_slicer_proc.start()

        online_slicer_proc.join()
        self.__log("Finished")

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
