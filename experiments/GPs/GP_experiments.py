import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from librosa import note_to_hz as hz
from datetime import datetime
import scipy.io.wavfile as wavf
from sklearn.gaussian_process.kernels import Matern

import helper
from least_squares import opt_amplitude
# from GP_models.onset_detection import detected_samples


number_samples = 2000
multiplier = 2


# Training data
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/piano_E4_330.wav'
sample_rate, data = wav.read(wav_file)
data = data[:]

data1 = data[5000:5700]

audio_duration = len(data1)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data1))

# Test data
T_test = np.linspace(0, audio_duration*multiplier, number_samples)

audio_duration0 = len(data)/sample_rate
time_samples0 = np.linspace(0, audio_duration0, len(data))

# Experimentation area

cov_dict = {}
sample_length = 700
sample_rate = 44100
M = 9
transition = 0.465
v = 2.37
sigma_f = 1/500000
sigma_n = 0.01
time_samples = np.linspace(0, sample_length/sample_rate, sample_length)
max_d = 50


print("state 5 ", -helper.stable_nlml(time_samples, data1,  M=9,
      normalised=False, f=[330], T=0.465, v=2.37, sigma_f=1/500000, sigma_n=0.01))
print("state 6 ", -helper.stable_nlml(time_samples, data1,  M=9,
      normalised=False, f=[330, 294], T=0.465, v=2.37, sigma_f=1/500000, sigma_n=0.01))
# print("state 7 ", -helper.stable_nlml(time_samples, data1,  M=9,
#       normalised=False, f=[440, 349, 262, 175], T=0.465, v=2.37, sigma_f=1/500000, sigma_n=0.01))


# --------------------------------------------------------------------------------------------------
# Implementation: Section 1— Model selection: Analysing audio data
# # # --------------------------------------------------------------------------------------------------
# helper.plot_audio(time_samples0, data, show=True,
#                   title="1000 samples of single A4 (440 Hz) piano note")
# helper.plot_audio(time_samples, data1, show=True,
#                   title="1000 samples of single A4 (440 Hz) piano note")
# helper.plot_fft(data1, sample_rate, colour='r', power_spectrum=True,
#                 title="PSD of 1000 samples of single A4 (440 Hz) piano note - Hanning window")
# plt.vlines(440, ymin=-
#            4, ymax=4, colors='pink', zorder=2)
# plt.vlines(440/3, ymin=-
#            4, ymax=4, colors='pink', zorder=2)
# plt.show()
# helper.plot_fft(data1, sample_rate, colour='r', power_spectrum=False,
#                 title="FFT of 1000 samples of single A4 (440 Hz) piano note - Hanning window")
# plt.vlines(440, ymin=-
#            3, ymax=2, colors='pink', zorder=2)
# plt.vlines(440/3, ymin=-
#            3, ymax=2, colors='pink', zorder=2)
# plt.show()
# TODO need to add lines to show the partials/ fundamental frequency


# --------------------------------------------------------------------------------------------------
# Plotting Covariance function
# --------------------------------------------------------------------------------------------------

# X = np.arange(-0.1, 1, 0.01)
# zeros = np.zeros(X.shape)
# cov = helper.SM_kernel(X, X, sigma_f=1/500000)
# kernel = cov[0]
# plt.plot(X+0.1, kernel)
# plt.show()
# # --------------------------------------------------------------------------------------------------
# Least squares optimise amplitude
# --------------------------------------------------------------------------------------------------
# a = opt_amplitude(data, f=[220, 349], M=3, sigma_f=5, show=True, T=2.5)[0]
# print(a)
# plt.show()

# --------------------------------------------------------------------------------------------------
# Plotting prior
# --------------------------------------------------------------------------------------------------

# X = np.linspace(0.0, audio_duration, len(data1)).reshape(-1, 1)
# T_test = T_test.reshape(-1, 1)
# # # cov = helper.RBF_kernel(X, T_test, l=1, sigma_f=1.0)
# cov = helper.SM_kernel(X, X, M=13, sigma_f=1/500000, f=[440], T=2.5, v=5)
# helper.plot_kernel_matrix(cov)
# mu = np.zeros(len(X))
# # Draw three samples from the prior
# samples = np.random.multivariate_normal(mu.ravel(), cov, 4)
# helper.plot_gp(mu, cov, X, samples=samples)
# plt.show()

# --------------------------------------------------------------------------------------------------
# Plotting posterior
# --------------------------------------------------------------------------------------------------

# mu_s, cov_s = helper.posterior(
#     T_test, time_samples, data,  M=15, sigma_f=1/500, f=[440], T=2, v=5)
# samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 1)
# helper.plot_gp(mu_s, cov_s, T_test, T_train=time_samples,
#                Y_train=data, samples=samples)
# plt.show()

# Write the predicted mean to a wave file
# mu_s = mu_s.ravel()
# data = np.array(mu_s)
# out_f = 'out.wav'
# artifical_sample_rate = number_samples/(audio_duration * multiplier)
# wavf.write(out_f, artifical_sample_rate, data)

# --------------------------------------------------------------------------------------------------
# Calculating likelihoods
# --------------------------------------------------------------------------------------------------
# t1 = datetime.utcnow().timestamp()
# print("LML for f=[294]", -helper.stable_nlml(
#     time_samples, data, M=12,  f=[349]))
# t2 = datetime.utcnow().timestamp()
# print("time difference is ", t2-t1)
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
# data1 = data1[2000:4000]

# audio_duration = len(data1)/sample_rate
# time_samples = np.linspace(0, audio_duration, len(data1))

# wav_file2 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/G_196_D_293.wav'
# sample_rate2, data2 = wav.read(wav_file2)
# # Truncate data to 2000 samples long
# data2 = data2[2000:4000]

# wav_file3 = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/D_min_triad.wav'
# sample_rate3, data3 = wav.read(wav_file3)
# # Truncate data to 2000 samples long
# data3 = data3[2000:4000]

# nlml = np.array([])
# for i in [data1, data2, data3]:
#     nlml = np.append(nlml, (helper.stable_nlml(time_samples, i, f=[
#         196, 233, 277, 330], M=10, sigma_f=2.72,  amplitude=0.000205)))

# print(nlml)


# --------------------------------------------------------------------------------------------------
# Optimising parameters (WIP) TODO jc to optimise the values of B (inharmonicity values)
# --------------------------------------------------------------------------------------------------
# res = minimize(helper.nlml_fn(time_samples, helper.power_normalise(data), f=[349], M=10), [1, 1],
#                bounds=((0, None), (0, None)),
#                method='L-BFGS-B')
# # res = scipy.optimize(helper.stable_nlml(time_samples, data), f=[400])
# sigma_f_opt, amplitude_opt = res.x
# print(sigma_f_opt, amplitude_opt)

# B_opt = helper.golden_section_B(
#     0.0002, 0.0007, time_samples, data, M=10, sigma_f=2.72, f=[1329], amplitude=0.000205)
# print(B_opt)
# f_opt = helper.golden_section_B(
#     400, 500, time_samples, data, M=14, sigma_f=1e-5, f=[494], integer_search=True)
# print(T_opt, v_opt)
