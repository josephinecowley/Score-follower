import numpy as np
from scipy.fft import fft
from scipy.fft import fftfreq
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

v = 2
k = 3.5
# This has been tweaked to experimentally add weights according to m


def MoG_spectral_kernel(x1, x2, M=3,  sigma=0.00001, frequencies=[440, 500]):
    """MoG spectral kernel
    M is the number of partials or harmonics for each note source
    TODO add weights k
    TODO add variance changes across Qs and Ms
    """
    cosine_series = 0
    t1 = x1/sample_rate
    t2 = x2/sample_rate
    for fundamental_frequency in frequencies:
        for m in range(M):
            cosine_series += k * (1/(1+((1/fundamental_frequency)*(m+1))**v))*np.cos((m+1) * 2 * np.pi *
                                                                                     fundamental_frequency * np.linalg.norm(t1 - t2))
    return np.exp(-(sigma**2) * 2*np.pi**2 * np.linalg.norm(t1 - t2)**2) * cosine_series
# made times


def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) *
        np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/tuner_440.wav'

# Read a WAV file
sample_rate, data = wav.read(wav_file)
data = data[:1000]
length = data.shape[0] / sample_rate

# Plotting the wave form in the time domain
time = np.linspace(0., length, data.shape[0])
plt.plot(time, data, label=" channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")


# Convert audio data to frequency domain
fft_data = fft(data)
N = len(data)
normalise = N/2


cov_function = []
# Plot covariance
for i in range(data.shape[0]):
    cov_function.append(MoG_spectral_kernel(
        i, 0, M=4, sigma=1e-10, frequencies=[440]))

plt.plot(time, cov_function[:], label=" channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()


# Get the frequency components of the spectrum
frequency_axis = fftfreq(N, d=1.0/sample_rate)
# Squared because the kernel is the power spectrum, so need to plot the square
norm_amplitude = np.abs(fft_data)**2
# Plot the results
plt.plot(frequency_axis, norm_amplitude, 'r')
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude')
plt.title('Spectrum')


# Convert audio data to frequency domain
fft_data = fft(cov_function)
N = len(cov_function)
normalise = N/2

# Get the frequency components of the spectrum
frequency_axis = fftfreq(N, d=1/sample_rate)
norm_amplitude = np.abs(fft_data)
# Plot the results
plt.plot(frequency_axis, norm_amplitude)
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude')
plt.title('Spectrum')
plt.show()

# Let's create hand tuned frequency spectrum of covariance Sx
# M = 8
# sigma = 0.7
# frequency = 440
# v = 1e-10
# x_values = np.linspace(-10000, 10000, 5000)
# y_values = gaussian(x_values, 440, 1) + gaussian(x_values, -440, 1)
# k = []

# extra = gaussian(x_values, 10000, sigma)/100

# for m in range(M):
#     # Initiate k
#     k_m = 1/(1 + ((m+1)/frequency)**v)
#     print((1 + ((m+1)/frequency)**v))
#     y_values += k_m * (gaussian(x_values, frequency*(m+1), sigma) +
#                        gaussian(x_values, -frequency*(m+1), sigma))

# plt.plot(x_values, y_values)
# plt.plot(x_values, extra)

# plt.show()
