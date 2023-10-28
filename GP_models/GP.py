import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

import helper

# Training data
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/A_440_piano.wav'
sample_rate, data = wav.read(wav_file)
# Truncate data to 700 samples long
data = data[500:900]
audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))

# Test data
T_test = np.linspace(0, audio_duration*1.5, 400)

# ---------------------
# Plotting out frequency spectrum of kernel power spectrum directly
# ---------------------

# helper.plot_fft(data, sample_rate)
# output, f_spectrum = helper.return_kernel_spectrum(
#     f=[440], M=8, sigma_f=20,  max_freq=20000, no_samples=len(data))  # sigma_f allows the gps to lookthe same height more or less....
# # sqrt because it was the power spectrum we created
# plt.plot(f_spectrum, 69*np.sqrt(output))
# plt.show()


# ---------------------
# Plotting audio and covariance samples
# ---------------------

# helper.plot_audio(time_samples, data)
# helper.return_SM_kernel(time_samples, M=5, f=[440], sigma_f=1e-1, show=True)

# ---------------------
# Plotting posterior
# ---------------------

mu_s, cov_s = helper.posterior(
    T_test, time_samples, data, M=8, f=[440], sigma_f=20)

mu = np.zeros(len(T_test))
helper.plot_gp(mu_s, cov_s, T_test, T_train=time_samples,
               Y_train=data, samples=4)
