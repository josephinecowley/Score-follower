import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.fft import fftfreq
from scipy.signal.windows import hann
import scipy.io.wavfile as wavf
import inharmonicity


def psd(audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
    w = hann(len(audio_samples))
    psd = np.abs(fft(audio_samples*w))**2 / len(audio_samples)
    frequency_axis = fftfreq(len(audio_samples), d=1.0/sample_rate)
    return psd[:(len(audio_samples)//8)], frequency_axis[:(len(audio_samples)//8)]


def phi_row(frequency_inputs, fundamental, M=12, std_dev=10, v=None, T=None, B=None, missing_fund: int = 0):
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
        # TODO set to 2 for missing fund
        inharmonicity_const = np.sqrt((1 + B * (m+1+missing_fund)**2))
        frequency = (m+1+missing_fund) * fundamental * inharmonicity_const
        filter = 1/(1 + (T*(m+1+missing_fund))**v)
        positive += filter * np.exp(-np.power((frequency_inputs -
                                              frequency) / std_dev, 2.0) / 2)
    return 1.0 / (np.sqrt(2.0 * np.pi) * std_dev) * positive


def phi_matrix(frequency_axis, f, M=10, sigma_f=10, T=None, v=None, B=None, missing_fund: int = 0):
    phi = np.zeros((len(f), len(frequency_axis)))
    for i, fundamental_frequency in enumerate(f):
        phi[i] = phi_row(
            frequency_axis, fundamental_frequency, M=M, std_dev=sigma_f, T=T, v=v, B=B, missing_fund=missing_fund)
    return phi.T


def opt_amplitude(data, sample_rate=44100, f=[440], M=10, sigma_f=10, T=None, v=None, B=None, show=False, missing_fund: int = 0):
    """
    Returns a flattened array a of amplitudes corresponding to each note source
    """
    psd_data, frequency_axis = psd(data, sample_rate)
    psd_data = psd_data.reshape((-1, 1))
    phi = phi_matrix(frequency_axis, f=f, M=M, sigma_f=sigma_f,
                     v=v, T=T, B=B, missing_fund=missing_fund)
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
    return a.flatten(),  prediction, psd_data, frequency_axis, sq_res[0, 0]
