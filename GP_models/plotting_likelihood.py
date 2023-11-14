import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tqdm import tqdm

import helper


wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/piano_f4_349.wav'
sample_rate, data = wav.read(wav_file)
data = data[9600:10600]
audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))

x = np.arange(0, 40, 2)
# x = np.array([0.0004,  0.0005, 0.0006])
y = np.array([])

for i in tqdm(x):
    y = np.append(y, helper.stable_nlml(
        time_samples, data, M=16, sigma_f=5, f=[349], amplitude=i))
# y.append(helper.stable_nlml(time_samples, data, M=14,
#          f=[494], sigma_f=i))  # -> gave us 0.2 for C5
# y = np.array([-1484.9293329801926, -1421.9849767049432, -1395.4419835024084, -1378.2306646540849, -1365.4242215849604, -1355.1982185486286, -1346.6715758273397, -1339.350462098267, -
#   1332.929917453104, -1327.2082449652166, -1322.0450260427715, -1317.3385292201551, -1313.0126629283595, -1309.0090067759347, -1305.28171961054, -1301.794160191411, -1298.5165730341284])

print(-y)
plt.plot(x, -y)
plt.axvline(x=349, color='pink', linestyle='--')
plt.title("LM Likelihood function with varying amplitude for piano 349Hz")
plt.show()
# print(helper.stable_nlml(time_samples, data, f=[330]))
