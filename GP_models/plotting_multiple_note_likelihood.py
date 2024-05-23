import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tqdm import tqdm

import helper
# This takes way too long

wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/G_196_D_293.wav'
sample_rate, data = wav.read(wav_file)
data = data[9600:9620]
audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))

x = np.arange(200, 600, 30)
y = np.zeros((300, 300))

for i, frequency1 in tqdm(enumerate(x)):
    for j, frequency2 in enumerate(x):
        y[i, j] = helper.stable_nlml(time_samples, data, f=[
                                     frequency1, frequency2])
    # y.append(helper.stable_nlml(time_samples, data, M=14, f=[494], sigma_f=i))-> gave us 0.2 for C5
# Convert to simple likelihoods
print(y)
plt.imshow(y, cmap='coolwarm', interpolation='nearest')
plt.title("Covariance Matrix Heatmap")
plt.colorbar()
plt.show()
# plt.axvline(x=293, color='pink', linestyle='--')
# print(helper.stable_nlml(time_samples, data, f=[330]))
