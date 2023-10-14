import numpy as np
from scipy.fft import fft
from scipy.fft import fftfreq
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt



def create_sine_wave(name='sine.wav', sample_rate=44100, frequency=440, length=5):
    sample_rate = sample_rate
    frequency = frequency
    length = length
    name = name

    # Produces Audio-File
    t = np.linspace(0, length, sample_rate * length)
    y = np.sin(frequency * 2 * np.pi * t)

    wav.write(name, sample_rate, y)
    return name


wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/octave.wav'
# wav_file = create_sine_wave("Sine.wav", frequency=4440)

# Read a WAV file
sample_rate, data = wav.read(wav_file)

length = data.shape[0] / sample_rate

# Plotting the wave form in the time domain
time = np.linspace(0., length, data.shape[0])
plt.plot(time, data[:], label=" channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()


# Convert audio data to frequency domain
fft_data = fft(data)
N = len(data)
normalise = N/2

# Get the frequency components of the spectrum
frequency_axis = fftfreq(N, d=1.0/sample_rate)
norm_amplitude = np.abs(fft_data)/normalise
# Plot the results
plt.plot(frequency_axis, norm_amplitude)
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude')
plt.title('Spectrum')
plt.show()
