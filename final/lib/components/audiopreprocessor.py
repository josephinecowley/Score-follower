from ..mputils import write_list_to_queue

from ..eprint import eprint
from lib.sharedtypes import AudioFrame, AudioFrameQueue
from typing import Callable, Optional, Dict, List, Union
import sounddevice as sd
import multiprocessing as mp
import librosa  # type: ignore
import time
import numpy as np
import sys
import scipy.io.wavfile as wav
from GP_models import helper
from matplotlib import pyplot as plt

# import time


class Slicer:
    def __init__(
        self,
        wave_path: Optional[str],
        hop_length: int,
        frame_length: int,
        sample_rate: int,
        audio_frames_queue: AudioFrameQueue,
        sleep_compensation: float,
        max_duration: float,
    ):
        self.wave_path = wave_path
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.audio_frames_queue = audio_frames_queue
        self.sleep_compensation = sleep_compensation
        self.max_duration = max_duration

        self.__log("Initialised successfully")

    def start(self):
        if self.wave_path:

            self.__log("Starting using performance recording...")
            sample_rate, data = wav.read(self.wave_path)
            sample_indices = np.arange(0, len(data), self.hop_length)
            audio_stream = [data[index:index+self.frame_length]
                            for index in sample_indices]

            self.__sleep(
                self.hop_length, time.perf_counter())
            for audio_frame in audio_stream:
                pre_sleep_time = time.perf_counter()
                self.audio_frames_queue.put(audio_frame)
                # sleep for hop length
                self.__sleep(self.hop_length, pre_sleep_time)

            self.audio_frames_queue.put(None)  # end
            self.__log("Finished")

        # if self.wave_path:  # If htere is a wave path, we will not do live return
        # self.__log("Starting using performance recording...")
        # audio_stream = librosa.stream(
        #     path=self.wave_path,
        #     block_length=1,
        #     # TODO hop length parameter didn't seem to be working so this is a current fix
        #     frame_length=self.hop_length,
        #     hop_length=self.hop_length,
        #     mono=True,
        #     fill_value=0,
        #     duration=self.max_duration,
        # )

        # # before starting, sleep for frame_length
        # self.__sleep(
        #     self.hop_length, time.perf_counter())
        # # self.frame_length, time.perf_counter() + 0.2)

        # for audio_frame in audio_stream:
        #     pre_sleep_time = time.perf_counter()
        #     audio_frame = (audio_frame * 32767).astype(np.int16)
        #     self.audio_frames_queue.put(audio_frame[:self.frame_length])
        #     # sleep for hop length
        #     self.__sleep(self.hop_length, pre_sleep_time)

        # self.audio_frames_queue.put(None)  # end
        # self.__log("Finished")

        else:
            # If live listening
            self.__log("Starting to listen...")
            with sd.InputStream(callback=self.__callback, channels=1, samplerate=self.sample_rate, blocksize=self.hop_length):
                sd.sleep(self.max_duration * 1000)
            self.__log("Finished: duration max timeout")
            self.audio_frames_queue.put(None)  # end

    def __callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_frames_queue.put(indata[:self.frame_length])

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
        max_duration: float,
        # slicer
        wave_path: Optional[str],
        sleep_compensation: float,
        audio_frames_queue: AudioFrameQueue,

    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.max_duration = max_duration
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
            max_duration=self.max_duration,
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
