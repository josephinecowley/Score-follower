import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.fft import fft
from scipy.fft import fftfreq
from librosa import note_to_hz as hz
from scipy.signal.windows import hann, hamming

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
data = data[3000:10000]
audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))


def psd(audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
    w = hann(len(audio_samples))
    psd = np.abs(fft(audio_samples*w))**2 / len(audio_samples)
    frequency_axis = fftfreq(len(audio_samples), d=1.0/sample_rate)
    return psd[:(len(audio_samples)//8)], frequency_axis[:(len(audio_samples)//8)]


def phi_row(frequency_inputs, fundamental, M=12, std_dev=10, v=None, T=None, B=None):
    positive = np.zeros(len(frequency_inputs))
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
    return 1.0 / (np.sqrt(2.0 * np.pi) * std_dev) * positive


def phi_matrix(frequency_axis, f, M=12, sigma_f=3, T=None, v=None, B=None):
    phi = np.zeros((len(f), len(frequency_axis)))
    for i, fundamental_frequency in enumerate(f):
        phi[i] = phi_row(
            frequency_axis, fundamental_frequency, M=M, std_dev=sigma_f, T=T, v=v, B=B)
    return phi.T


def least_squares(data, sample_rate=44100, f=[440], M=10, sigma_f=5, T=None, v=None, B=None, show=False):
    """
    Returns an array a of amplitudes corresponding to each note source
    """
    psd_data, frequency_axis = psd(data, sample_rate)
    psd_data = psd_data.reshape((-1, 1))
    phi = phi_matrix(frequency_axis, f=f, M=M, sigma_f=sigma_f, v=v, T=T, B=B)
    inverse_factor = np.linalg.inv(phi.T @ phi)
    a = inverse_factor @ phi.T @ psd_data
    prediction = phi @ a
    res = prediction - psd_data
    sq_res = res.T @ res
    if show is True:
        plt.plot(frequency_axis, psd_data, label="PSD of audio")
        plt.plot(frequency_axis, prediction,
                 label="Kernel in frequency domain")
        plt.legend()
    return a,  prediction, psd_data, frequency_axis, sq_res[0, 0]


frequencies = hz(['G3', 'A#3', 'C#4', 'E4'])

a = least_squares(data, f=frequencies, M=6, T=2, show=True, sigma_f=2)[0]
print(a)
plt.show()