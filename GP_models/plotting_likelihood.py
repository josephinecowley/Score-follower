import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
import helper
from GP_models.onset_detection import detected_samples
from librosa import note_to_hz as hz
from least_squares import opt_amplitude


scale_freqs = hz([['C4'], ['D4'], ['E4'], ['F4'],
                 ['G4'], ['A4'], ['B4'], ['C5']])
beethoven_chords = [['E4', 'G#4', 'B4', 'E5', 'G#5'],
                    ['B#3', 'D#4', 'F#4', 'A4', 'D#5', 'F#5'], ['C#4', 'E4', 'G#4', 'C#5', 'D#5'], ['G#3', 'B#3', 'D#4', 'F#4', 'B#4', 'D#5']]
beethoven_freqs = [hz(notes) for notes in beethoven_chords]
# frequencies = hz(['G3', 'A#3', 'C#4', 'E4'])
frequencies = hz(['D4', 'F4', 'A4'])
print(beethoven_freqs[0])
# -------------------------------------------------#
# For individual notes
# -------------------------------------------------#
# wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/G_diminished.wav'
# sample_rate, data = wav.read(wav_file)
# data = data[3600:5600]

# audio_duration = len(data)/sample_rate
# time_samples = np.linspace(0, audio_duration, len(data))
# -------------------------------------------------#
# onset detection of multiple notes
# # # -------------------------------------------------#
sample_data, sample_rate = detected_samples(
    '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/D_min_triad.wav', sample_length=3000, offset=3000, show=True, delta=0.12)
# sample_data2, sample_rate2 = detected_samples(
#     '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/beethoven.wav', sample_length=2000, offset=1000)
plt.show()

# -------------------------------------------------#
# Get rid of noise notes for clarity of results
# -------------------------------------------------#
# indices_to_delete = [5]  # List of indices to delete
# sample_data = [element for index, element in enumerate(
#     sample_data) if index not in indices_to_delete]
# print(len(sample_data))
# -------------------------------------------------#
#
# -------------------------------------------------#
# sample_data = sample_data + sample_data2[:3]

# x = np.arange(0, 180, 20)
# x = np.linspace(0, 600, 35)
time_samples = np.linspace(
    0, len(sample_data[0])/sample_rate, len(sample_data[0]))
# helper.plot_fft(helper.power_normalise(sample_data[]))

# print(-helper.stable_nlml(
#     time_samples, helper.power_normalise(data), M=10, sigma_f=300.705, f=[351.7], amplitude=200.5))
# for i in range(4):
#     for j in range(4):
#         y[i, j] = -helper.stable_nlml(time_samples,
#                                       sample_data[i], M=9, sigma_f=0.705, f=beethoven_freqs[j], amplitude=2.5)
# print(y)

# # optimal values for 349 hz in scale, f=351.7 M=10, sigma_f = 2.705, amplitude = 0.000205 LML= 5171
# # optimal values for 349 Hz normalised to 50 per 2000 samples, is: sigma=18.5, M=9; sigma=0.0002; f=[344.5]

a = opt_amplitude(sample_data[0], f=frequencies, T=2, show=True)[0]
plt.show()
print('a is', a)
print(-helper.stable_nlml(time_samples, sample_data[0],
      f=frequencies, amplitude=None))
# for i in range(y.shape[0]):
#     plt.plot(y[i], marker='x')
#     plt.show()
# plt.plot(-y, marker='x')
# plt.show()
# plt.axvline(x=349, color='pink', linestyle='--')
# plt.title("LM Likelihood function with varying sigma for piano 349Hz")
# plt.show()
# print(helper.stable_nlml(time_samples, data, f=[330]))


# res = minimize(helper.nlml_fn(time_samples, sample_data[3], f=[int(hz('C4'))], M=10), [1, 1],
#                bounds=((0, None), (0, None)),
#                method='CG')
# # res = scipy.optimize(helper.stable_nlml(time_samples, data), f=[400])
# sigma_f_opt, amplitude_opt = res.x
# print(sigma_f_opt, amplitude_opt)
