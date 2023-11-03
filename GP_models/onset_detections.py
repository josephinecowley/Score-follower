import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import helper
import numpy as np
from librosa import note_to_hz as hz

scale_freqs = hz([['C4'], ['D4'], ['E4'], ['F4'],
                 ['G4'], ['A4'], ['B4'], ['C5']])
beethoven_notes = [['E4', 'G#4', 'B4', 'E5', 'G#5'],
                   ['B#3', 'D#4', 'F#4', 'A4', 'D#5', 'F#5'], ['C#4', 'E4', 'G#4', 'C#5', 'D#5'], ['G#3', 'B#3', 'D#4', 'F#4', 'B#4', 'D#5']]
beethoven_freqs = [hz(notes) for notes in beethoven_notes]
print(hz(['E4', 'G#4', 'B4', 'E5', 'G#5']))

# JC note fundamental -- or mark expected harmoncs (with a number)

wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/beethoven.wav'
sample_rate, data = wav.read(wav_file)
onset_times = librosa.onset.onset_detect(
    y=data, post_avg=5, wait=1,  sr=sample_rate, units='time')  # delta=0.15, wait=5, pre_avg=0, post_avg=5, pre_max=5, post_max=10,

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
    # sample_time_samples = np.linspace(
    #     0, 10000/sample_rate, len(sample_data[i]))
    helper.plot_fft(sample_data[i], sample_rate)
    helper.return_kernel_spectrum(
        f=beethoven_freqs[i], show=True, scalar=50)
    # print(helper.stable_nlml(sample_time_samples,
    #       sample_data[i], f=hz(['E4', 'G#4', 'B4', 'E5', 'G#5'])))
    plt.show()
