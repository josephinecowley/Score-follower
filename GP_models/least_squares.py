import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.fft import fft
from scipy.fft import fftfreq
from librosa import note_to_hz as hz

import scipy.io.wavfile as wavf
import inharmonicity

import helper

number_samples = 1500
multiplier = 1.5


fig = plt.figure()
axes = []

titles = ['piano_C4_262', 'piano_D4_294',
          'piano_E4_330', 'piano_f4_349', 'piano_G4_392', 'piano_A4_440', 'piano_B4_494', 'piano_C5_523']
frequencies = [262, 294, 330, 349, 392, 440, 494, 523]
# Single note
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/G_diminished.wav'
sample_rate, data = wav.read(wav_file)
data = data[9000:15000]
audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))

helper.plot_fft(data)
plt.show()
psd, frequency_axis = helper.psd(data, sample_rate)


def phi_row(frequency_inputs, fundamental, M=12, std_dev=10, v=None, T=None, B=None):
    positive = np.zeros(len(frequency_inputs))
    negative = np.zeros(len(frequency_inputs))
    if v is None:
        v = 2.37
    if T is None:
        T = 0.465
    if B is None:
        closest_key = min(inharmonicity.B.keys(), key=lambda key: abs(
            key - fundamental))
        B = inharmonicity.B[closest_key]
    for m in range(M):
        inharmonicity_const = np.sqrt((1 + B * (m+1)**2))
        frequency = (m+1) * fundamental * inharmonicity_const
        filter = 1/(1 + (T*(m+1))**v)
        positive += filter * np.exp(-np.power((frequency_inputs -
                                              frequency) / std_dev, 2.0) / 2)
        negative += filter * np.exp(-np.power((frequency_inputs +
                                              frequency) / std_dev, 2.0) / 2)
    return 1.0 / (np.sqrt(2.0 * np.pi) * std_dev) * (positive + negative)


# def kernel(frequency_axis, f=[440], M=12, sigma_f=5, amplitude=1, v=None, T=None, B=None):
#     kernel = np.zeros(len(frequency_axis))
#     if v is None:
#         v = 2.37
#     if T is None:
#         T = 0.465
#     amplitudes = np.ones(np.array(f).shape)
#     print(amplitudes)
#     amplitudes[0] = 20

#     for i, fundamental_frequency in enumerate(f):
#         if B is None:
#             closest_key = min(inharmonicity.B.keys(),
#                               key=lambda key: abs(key - fundamental_frequency))
#             B = inharmonicity.B[closest_key]
#         for m in range(M):
#             inharmonicity_const = np.sqrt((1 + B * (m+1)**2))
#             kernel += amplitudes[i] * 1/(1 + (T*(m+1))**v) * gaussian(frequency_axis,
#                                                                       (m+1)*fundamental_frequency * inharmonicity_const, sigma_f)
#     return kernel


# kern = kernel(frequency_axis, [392, 4000], sigma_f=20, amplitude=20)
# plt.plot(frequency_axis, kern)
# plt.show()
# print(len(kern), len(psd))
frequencies = hz(['G3', 'A#3', 'C#4', 'E4'])
# frequencies = [196, 293]
print(frequencies)
# frequencies = [440]


def phi_matrix(frequency_axis, frequencies, M=12, std_dev=3):
    phi = np.zeros((len(frequencies), len(frequency_axis)))

    for i, fundamental_frequency in enumerate(frequencies):
        phi[i] = phi_row(
            frequency_axis, fundamental_frequency, M=M, std_dev=std_dev)
    return phi.T


phi = phi_matrix(frequency_axis, frequencies, M=2, std_dev=5)
inverse_factor = np.linalg.inv(phi.T @ phi)
psd = psd.reshape((-1, 1))
a = inverse_factor @ phi.T @ psd
plt.plot(frequency_axis, phi@a)
plt.show()
# a = phi.T @ data
# print(data)
# a = inverse_factor @ (phi.T @ data)
# print(a)
# plt.plot(frequency_axis, phi @ a)
# plt.show()
# print((psd[3000]), phi[3000])
# residual = phi @ a - data
# sq_residual = residual.T @ residual
# print(sq_residual)
# a *= 50
# residual = phi @ a - data
# sq_residual = residual.T @ residual
# print(sq_residual)
