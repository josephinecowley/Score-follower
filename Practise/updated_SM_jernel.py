import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.fft import fftfreq
import scipy.io.wavfile as wav


def SM_function(x1, x2, periods=[1/440], partial_count=5, weights=np.array([1]*20), std_devs=np.array([0.0009]*20)):
    # Ignoring amplitudes of each source
    difference = x1 - x2
    covariance = 0
    for period in periods:
        for partial in range(partial_count):
            m = partial+1
            covariance += weights[m-1] * np.exp(-2 * np.pi**2 * std_devs[m-1]**2 * difference**2) * np.cos(
                2 * np.pi * (1/m) * period * difference)
    return covariance


# Plot the kernel function for increasing tau = x1 - x2
inputs = np.arange(0, 2000, 1)
cov = np.array([])
for i in inputs:
    cov = np.append(cov, SM_function(i, 0))
plt.plot(inputs, cov)
plt.show()


# Take real data
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/tuner_440.wav'
# Read a Wav file
sample_rate, data = wav.read(wav_file)
data = data[300:1300]


# Plot the FFT of the kernel function
fft_kernel_function = np.abs(fft(cov, norm="ortho"))
# Get the frequency components of the spectrum
SM_frequency_axis = fftfreq(len(fft_kernel_function), d=1.0/30)
# Plot the results


# Plot fft functions
fft_data = np.abs(fft(data))
# Get the frequency components of the spectrum
frequency_axis = fftfreq(len(fft_data), d=1.0/sample_rate)
# Plot the results
# plt.plot(frequency_axis, fft_data)
plt.plot(SM_frequency_axis, fft_kernel_function)
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude')
plt.title('Spectrum')
plt.show()
