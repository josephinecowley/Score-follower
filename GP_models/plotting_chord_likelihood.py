import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import helper
import numpy as np
from librosa import note_to_hz as hz
from tqdm import tqdm

# Import 3 different chords

wav_file1 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/G_diminished.wav'
sample_rate1, data1 = wav.read(wav_file1)
# Truncate data to 2000 samples long
data1 = data1[2000:4000]
data1 = np.array(data1)
print(len(data1))

audio_duration = len(data1)/sample_rate1
time_samples = np.linspace(0, audio_duration, len(data1))

wav_file2 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/G_196_D_293.wav'
sample_rate2, data2 = wav.read(wav_file2)
# Truncate data to 2000 samples long
data2 = data2[2000:4000]
data2 = np.array(data2)

wav_file3 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/D_min_triad.wav'
sample_rate3, data3 = wav.read(wav_file3)
# Truncate data to 2000 samples long
data3 = data3[2000:4000]
data3 = np.array(data3)


# Import 4 different beethoven chords

beethoven_chords = [['E4', 'G#4', 'B4', 'E5', 'G#5'],
                    ['B#3', 'D#4', 'F#4', 'A4', 'D#5', 'F#5'], ['C#4', 'E4', 'G#4', 'C#5', 'D#5'], ['G#3', 'B#3', 'D#4', 'F#4', 'B#4', 'D#5']]
beethoven_freqs = [hz(notes) for notes in beethoven_chords]

wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/beethoven.wav'
sample_rate, data = wav.read(wav_file)
data = data[:]
onset_times = librosa.onset.onset_detect(
    y=data, post_avg=5, wait=1,  sr=sample_rate, units='time')  # delta=0.15, wait=5, pre_avg=0, post_avg=5, pre_max=5, post_max=10,

audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))
onset_time_samples = onset_times * sample_rate


helper.plot_audio(time_samples, data)
for time in onset_times:
    plt.vlines(time+2000/sample_rate, ymin=-2, ymax=2, colors='blue', zorder=2)
    plt.vlines(time+9000/sample_rate, ymin=-2, ymax=2, colors='pink', zorder=2)
plt.show()


sample_data = []

# Set the sample data for each chord
for i in range(3):
    sample_data.append(
        data[int(onset_time_samples[i] + 2000):int(onset_time_samples[i])+4000])
# add the different audio samples tgt
sample_data.append(data1)
sample_data.append(data2)
sample_data.append(data3)

sample_duration = len(sample_data[0])/sample_rate
time_samples = np.linspace(0, sample_duration, len(sample_data[0]))

nlml = np.array([])

# print(helper.stable_nlml(time_samples, sample_data[4], f=[
#     196, 233, 277, 330], M=10, sigma_f=2.72,  amplitude=0.000205))
for i in sample_data:
    nlml = np.append(nlml, (helper.stable_nlml(time_samples, i, f=[
        196, 233, 277, 330], M=10, sigma_f=2.72,  amplitude=0.000205)))

print(-nlml)
# # results: positive lml: [-3800.58974232 -2222.68928713 -8143.22993378  1606.44941861]
