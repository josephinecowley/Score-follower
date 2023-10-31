import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import helper
import numpy as np


wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/scale.wav'
sample_rate, data = wav.read(wav_file)
onset_times = librosa.onset.onset_detect(
    y=data, sr=sample_rate, wait=0.01, pre_avg=1, post_avg=1, pre_max=1, post_max=1, units='time')

audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))
onset_time_samples = onset_times * sample_rate

sample_data = []

helper.plot_audio(time_samples, data)
for time in onset_times:
    plt.vlines(time, ymin=-2, ymax=2, colors='blue', zorder=2)
plt.show()

for i in range(len(onset_time_samples)):
    print(int(onset_time_samples[i]))
    sample_data.append(data[int(onset_time_samples[i]):int(onset_time_samples[i])+10000])
    helper.plot_fft(sample_data[i], sample_rate)
    plt.show()
