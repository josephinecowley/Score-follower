import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tqdm import tqdm
import helper
# NOTE despite vectorising the m loop, we see that it is in fact slower — perhaps come bac to this later and see it there is any other way

wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/piano_F4_349.wav'
sample_rate, data = wav.read(wav_file)
data = data[3600:5600]


audio_duration = len(data)/sample_rate
time_samples = np.linspace(0, audio_duration, len(data))

X1 = time_samples.reshape(-1, 1)
X2 = time_samples.reshape(-1, 1)

cosine_series = np.zeros((X1.shape[0], X2.shape[0]))

X1_1D = X1.flatten()
X2_1D = X2.flatten()

v = 2.37
T = 0.465
B = 0.000046

M = 100
fundamental_frequency = 440
m = np.arange(1, M + 1)
m_reshape = m[:, np.newaxis, np.newaxis]

# Precompute constants
inharmonicity_const = np.sqrt(1 + B * m**2)
cosine_arg = 2 * np.pi * \
    inharmonicity_const[:, np.newaxis, np.newaxis] * \
    m_reshape * fundamental_frequency

# Compute the cosine series element-wise and then sum along the appropriate axis
cosine_series = np.sum(1 / (1 + (T * m[:, np.newaxis, np.newaxis])**v) *
                       np.cos(cosine_arg *
                              (X1_1D[np.newaxis, :, np.newaxis] - X2_1D[np.newaxis, np.newaxis, :])),
                       axis=0)

print(cosine_series)
