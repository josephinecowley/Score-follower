import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.fft import fft
from scipy.fft import fftfreq

import scipy.io.wavfile as wavf

import helper

number_samples = 1500
multiplier = 1.5


fig = plt.figure()
axes = []

titles = ['piano_C4_262', 'piano_D4_294',
          'piano_E4_330', 'piano_f4_349', 'piano_G4_392', 'piano_A4_440', 'piano_B4_494', 'piano_C5_523']
frequencies = [262, 294, 330, 349, 392, 440, 494, 523]
# Single note
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/piano_B4_494.wav'
sample_rate, data = wav.read(wav_file)
data = data[:]
audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))
print("length of data sample", len(data))
helper.plot_fft(data, sample_rate,
                title='Spectrum of piano note 494 Hz')
helper.return_kernel_spectrum(
    f=[494], M=9, amplitude=59, sigma_f=10, T=0.465, v=2.37, max_freq=5000)

# # multiple notes:``
# for i in range(len(titles)):
#     wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/' + \
#         titles[i]+'.wav'
#     sample_rate, data = wav.read(wav_file)
#     data = data[:]
#     print(len(data))
#     fft_data = abs(fft(data, norm="ortho"))
#     frequency_axis = fftfreq(len(data), d=1.0/sample_rate)
#     ax = fig.add_subplot(2, 4, i + 1)
#     axes.append(ax)
#     ax.plot(frequency_axis[:(len(data)//8)], fft_data[:(len(data)//8)], 'r')
#     ax.set_title('Subplot {}'.format(titles[i]))

#     # Plot kernel sqrt power spectrum
#     output, f_spectrum = helper.return_kernel_spectrum(
#         f=[frequencies[i]], M=8, sigma_f=10,  max_freq=5500, no_samples=len(data))
#     plt.plot(f_spectrum, 17*np.sqrt(output))

plt.show()
