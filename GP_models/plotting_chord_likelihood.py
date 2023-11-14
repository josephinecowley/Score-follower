import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import helper
import numpy as np
from librosa import note_to_hz as hz
from tqdm import tqdm


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
print(beethoven_freqs[0])

# Set the sample data for each chord
for i in range(len(onset_time_samples)):
    sample_data.append(
        data[int(onset_time_samples[i] + 2000):int(onset_time_samples[i])+3000])
    plt.show()

y = np.array([])
sample_duration = len(sample_data[0])/sample_rate
time_samples = np.linspace(0, sample_duration, len(sample_data[0]))
for i in tqdm(beethoven_freqs):
    # print(i)
    y = np.append(y, helper.stable_nlml(
        time_samples, sample_data[0], M=16, sigma_f=5, f=i, amplitude=25))
print(-y)
# results: positive lml: [-3800.58974232 -2222.68928713 -8143.22993378  1606.44941861]
