import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

import helper

# ---------------------
# Training data: 'times_samples' and 'data'
# ---------------------

# Wav file method
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/tuner_440.wav'

# Read a Wav file
sample_rate, data = wav.read(wav_file)

# Truncate data to 700 samples long
data = data[100:1000]
audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))

# helper.plot_fft(data, sample_rate, colour='k')

# ---------------------
# Plotting out frequency spectrum of kernel directly
# ---------------------

helper.plot_fft(data, sample_rate)
output, f_spectrum = helper.return_kernel_spectrum(
    f=[440], M=8, sigma_f=30,  max_freq=20000, no_samples=len(data))  # sigma_f allows the gps to lookthe same height more or less....
# sqrt because it was the power spectrum we created
plt.plot(f_spectrum, 9*np.sqrt(output))
plt.show()

# helper.plot_fft(data, sample_rate)


# ---------------------
# Test data: 'T_test'
# ---------------------

# Test data
# T_test = np.linspace(0, audio_duration*1.5, 400)


# ---------------------
# Test data: 'T_test'
# ---------------------

# helper.plot_audio(time_samples, data)
# helper.return_SM_kernel(time_samples, M=5, f=[440], sigma_f=1e-1, show=True)
# cov = helper.return_SM_kernel_matrix(T_test, M=5, f=[440], sigma_f=1e-2)

# mu = np.zeros(len(T_test))
# helper.plot_gp(mu, cov, T_test, T_train=time_samples, Y_train=data, samples=2)
