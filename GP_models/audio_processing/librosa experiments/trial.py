import lib.sharedtypes as sharedtypes
from lib.sharedtypes import ExtractedFeatureQueue
import lib.sharedtypes as sharedtypes
import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
import multiprocessing as mp


slice_queue: ExtractedFeatureQueue = mp.Queue()
audio_stream = librosa.stream(
    "/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/GP_models/audio_processing/librosa experiments/experiment.wav",
    block_length=100,
    frame_length=2000,
    hop_length=20000,
    mono=True,
    fill_value=0,
    duration=10,
)
time.sleep(0.5)
for s in audio_stream:
    time.sleep(0.5)
    slice_queue.put(s)
slice_queue.put(None)
print(slice_queue)
