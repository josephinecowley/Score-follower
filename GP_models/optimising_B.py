import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import helper
import numpy as np
from librosa import note_to_hz as hz
from tqdm import tqdm
import scipy
from scipy.optimize import minimize

# Training data
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/C4#_scale.wav'
sample_rate, data = wav.read(wav_file)
data = data[:]
onset_times = librosa.onset.onset_detect(
    y=data, post_avg=5, wait=1,  sr=sample_rate, units='time', delta=0.1)  # delta=0.15, wait=5, pre_avg=0, post_avg=5, pre_max=5, post_max=10,

audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))
onset_time_samples = onset_times * sample_rate

sample_data = []
helper.plot_audio(time_samples, data)
for i, time in enumerate(onset_times):
    plt.vlines(time+2000/sample_rate, ymin=-2, ymax=2, colors='blue', zorder=2)
    sample_data.append(
        data[int(onset_time_samples[i] + 2000):int(onset_time_samples[i])+3000])
plt.show()

sample_audio_duration = len(sample_data[0])/sample_rate
sample_time_samples = np.linspace(
    0, sample_audio_duration, len(sample_data[0]))
# WIP
