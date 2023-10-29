import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import scipy.io.wavfile as wavf

import helper

number_samples = 1500
multiplier = 1.5


# Training data
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/A_440_piano.wav'
sample_rate, data = wav.read(wav_file)
# Truncate data to 700 samples long
data = data[500:2000]
audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))

# Test data
T_test = np.linspace(0, audio_duration*multiplier, number_samples)

# --------------------------------------------------------------------------------------------------
# Plotting out frequency spectrum of kernel power spectrum directly
# --------------------------------------------------------------------------------------------------

helper.plot_fft(data, sample_rate, colour='r')
output, f_spectrum = helper.return_kernel_spectrum(
    f=[440], M=8, sigma_f=20,  max_freq=5500, no_samples=len(data))  # sigma_f allows the gps to lookthe same height more or less....
# sqrt because it was the power spectrum we created
plt.plot(f_spectrum, 69*np.sqrt(output))
plt.show()


# --------------------------------------------------------------------------------------------------
# Plotting audio and covariance samples
# --------------------------------------------------------------------------------------------------

# helper.plot_audio(time_samples, data)
# helper.return_SM_kernel(time_samples, M=5, f=[440], sigma_f=1e-1, show=True)

# --------------------------------------------------------------------------------------------------
# Plotting posterior
# --------------------------------------------------------------------------------------------------

# mu_s, cov_s = helper.posterior(
#     T_test, time_samples, data, M=8, f=[440], sigma_f=1, sigma_y=0.0005)


# helper.plot_gp(mu_s, cov_s, T_test, T_train=time_samples,
#                Y_train=data, samples=2)

# # Write the predicted mean to a wave file
# mu_s = mu_s.ravel()
# data = np.array(mu_s)
# out_f = 'out.wav'
# artifical_sample_rate = number_samples/(audio_duration * multiplier)
# wavf.write(out_f, artifical_sample_rate, data)

# --------------------------------------------------------------------------------------------------
# Calculating likelihoods
# --------------------------------------------------------------------------------------------------

# print(helper.stable_nlml(time_samples, data, f=[196, 233, 277, 311]))

# --------------------------------------------------------------------------------------------------
# Comparing likelihoods of single notes
# --------------------------------------------------------------------------------------------------
# wav_file1 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/tuner_440.wav'
# sample_rate1, data1 = wav.read(wav_file1)
# # Truncate data to 900 samples long
# data1 = data1[500:900]

# audio_duration = len(data1)/sample_rate
# time_samples = np.linspace(0, audio_duration, len(data1))

# wav_file2 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/D#4_311.wav'
# sample_rate2, data2 = wav.read(wav_file2)
# # Truncate data to 900 samples long
# data2 = data2[500:900]

# wav_file3 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/f4_349.wav'
# sample_rate3, data3 = wav.read(wav_file3)
# # Truncate data to 900 samples long
# data3 = data3[500:900]

# for i in [data1, data2, data3]:
#     print(helper.stable_nlml(time_samples, i, f=[311]))

# --------------------------------------------------------------------------------------------------
# Comparing likelihoods of chords -- NOte that the length of time sample is significant.
#
# concerning as results for this were:
# -2111.231953020822
# -1432.5814599823302
# -2170.7988116328384
# --------------------------------------------------------------------------------------------------

# wav_file1 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/G_diminished.wav'
# sample_rate1, data1 = wav.read(wav_file1)
# # Truncate data to 2000 samples long
# data1 = data1[500:2000]

# audio_duration = len(data1)/sample_rate
# time_samples = np.linspace(0, audio_duration, len(data1))

# wav_file2 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/G_196_D_293.wav'
# sample_rate2, data2 = wav.read(wav_file2)
# # Truncate data to 2000 samples long
# data2 = data2[500:2000]

# wav_file3 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/D_min_triad.wav'
# sample_rate3, data3 = wav.read(wav_file3)
# # Truncate data to 2000 samples long
# data3 = data3[500:2000]

# for i in [data1, data2, data3]:
#     print(helper.stable_nlml(time_samples, i, f=[196, 233, 277, 330]))


# --------------------------------------------------------------------------------------------------
# Optimising parameters (WIP)
# --------------------------------------------------------------------------------------------------
# res = minimize(helper.nlml_fn(time_samples, data), [300, 1, 0.005],
#                bounds=((1e-5, None), (1e-5, None)),
#                method='L-BFGS-B')
# f_opt = helper.golden_section(20, 800, time_samples, data)
# f_opt, sigma_f_opt, sigma_n_opt = res.x
