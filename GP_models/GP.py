import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize

import scipy.io.wavfile as wavf
from sklearn.gaussian_process.kernels import Matern

import helper

number_samples = 150
multiplier = 1


# Training data
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/piano_f4_349.wav'
sample_rate, data = wav.read(wav_file)
# Truncate data to 700 samples long
data = data[3600:4600]
audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))


# Test data
T_test = np.linspace(0, audio_duration*multiplier, number_samples)

# --------------------------------------------------------------------------------------------------
# Plotting out frequency spectrum of kernel power spectrum directly
# --------------------------------------------------------------------------------------------------
# helper.plot_audio(time_samples, data, show=True,
#                   title="2000 samples")
# helper.plot_fft(data, sample_rate, colour='r',
#                 title="FFT of 2000 samples - Hanning window")
# plt.show()


# output, f_spectrum = helper.return_kernel_spectrum(
#     f=[330], M=8, sigma_f=10,  max_freq=5500, no_samples=len(data))  # sigma_f allows the gps to lookthe same height more or less....
# # sqrt because it was the power spectrum we created
# plt.plot(f_spectrum, 17*np.sqrt(output))
# plt.show()


# --------------------------------------------------------------------------------------------------
# Plotting audio and covariance samples
# --------------------------------------------------------------------------------------------------

# # helper.plot_audio(time_samples, data)
# X = np.arange(-0.1, 0.1, 0.0001)
# helper.return_SM_kernel(
#     X, M=3, f=[1760], sigma_f=5, show=True, amplitude=5, title="f=1760 Hz; M=3; sigma=5; amplitude=5")
# helper.return_SM_kernel(
#     X, M=16, f=[1760], sigma_f=5, show=True, amplitude=5, title="f=1760 Hz; M=16; sigma=5; amplitude=5")
# helper.return_SM_kernel(
#     X, M=3, f=[1760], sigma_f=20, show=True, amplitude=5, title="f=1760 Hz; M=3; sigma=20; amplitude=5")
# helper.return_SM_kernel(
#     X, M=3, f=[1760], sigma_f=5, show=True, amplitude=25, title="f=1760 Hz; M=3; sigma=5; amplitude=25")


# --------------------------------------------------------------------------------------------------
# Plotting prior
# --------------------------------------------------------------------------------------------------
# Finite number of points
title = "f=[261,1760] Hz; M=16; sigma=20; amplitude=1"
X = np.arange(-0.005, 0.005, 0.00005).reshape(-1, 1)
zero = np.zeros(X.shape)
# cov = helper.RBF_kernel(X, X, l=1, sigma_f=1.0)
cov = helper.improved_SM_kernel(
    X, X, f=[2170], M=16, sigma_f=20, amplitude=5)
mu = np.zeros(len(X))
# Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 1)
helper.plot_gp(mu, cov, X, samples=samples,
               title=title)
plt.show()

# --------------------------------------------------------------------------------------------------
# Plotting posterior
# --------------------------------------------------------------------------------------------------

# mu_s, cov_s = helper.posterior(
#     T_test, time_samples, data, M=14, f=[330], sigma_f=1, sigma_y=0.0005)


# helper.plot_gp(mu_s, cov_s, T_test, T_train=time_samples,
#                Y_train=data, samples=2)
# plt.show()

# # Write the predicted mean to a wave file
# mu_s = mu_s.ravel()
# data = np.array(mu_s)
# out_f = 'out.wav'
# artifical_sample_rate = number_samples/(audio_duration * multiplier)
# wavf.write(out_f, artifical_sample_rate, data)

# --------------------------------------------------------------------------------------------------
# Calculating likelihoods
# --------------------------------------------------------------------------------------------------

# print("for f=349 (correct)", helper.stable_nlml(
#     time_samples, data, M=16, sigma_f=5, f=[349], amplitude=0.001))  # with 1000 samples, and amplitude = 0.001, this gives -3425.5
# print("for f=50 ()", helper.stable_nlml(
#     time_samples, data, f=[50], amplitude=0.001))
# print("for f=50 ()", helper.stable_nlml(
#     time_samples, data, f=[5], amplitude=0.001))

# print("for f=349 (correct)", helper.stable_nlml(
#     time_samples, data, M=16, sigma_f=5, f=[349], amplitude=25))  # with 1000 samples, and amplitude = 0.001, this gives -3425.5
# print("for f=50 ()", helper.stable_nlml(
#     time_samples, data, f=[50], amplitude=25))
# print("for f=50 ()", helper.stable_nlml(
#     time_samples, data, f=[5], amplitude=25))
# print("for f=623 (wrong)", helper.stable_nlml(time_samples, data, f=[623]))
# print("for f=423 (wrong)", helper.stable_nlml(time_samples, data, f=[423]))
# focus on the liklihood functions

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
# Optimising parameters (WIP) TODO jc to optimise the values of B (inharmonicity values)
# --------------------------------------------------------------------------------------------------
# res = minimize(helper.nlml_fn(time_samples, data, f=[494]), [1, 1],
#                bounds=((0, 3), (2.3, 2.4)),
#                method='L-BFGS-B')
# # res = scipy.optimize(helper.stable_nlml(time_samples, data), f=[400])
# T_opt, v_opt = res.x

# B_opt = helper.golden_section_B(
#     0.0002, 0.0007, time_samples, data, M=16, sigma_f=5, f=[349])
# print(B_opt)
# f_opt = helper.golden_section_B(
#     400, 500, time_samples, data, M=14, sigma_f=1e-5, f=[494], integer_search=True)
# print(T_opt, v_opt)
