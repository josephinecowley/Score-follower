
import numpy as np
import scipy.io.wavfile as wav
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular

name = "scale"
noise = 0.004


def MoG_spectral_kernel_matrix(X1, X2, M=20, sigma=10, frequencies=[440]):
    """
    Compute the MoG spectral kernel matrix between two sets of vectors.

    X1 and X2 are arrays of shape (n1, d) and (n2, d), where n1 and n2 are the
    numbers of vectors and d is the dimension of each vector.

    M is the number of partials or harmonics for each note source.
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    kernel_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            x1 = X1[i, :]
            x2 = X2[j, :]

            cosine_series = 0
            for fundamental_frequency in frequencies:
                for m in range(M):
                    cosine_series += np.cos((m + 1) * 2 * np.pi *
                                            fundamental_frequency * np.linalg.norm(x1 - x2))

            kernel_value = (1 / np.pi) * np.exp(-(sigma**2 / 2)
                                                * np.linalg.norm(x1 - x2)**2) * cosine_series
            kernel_matrix[i, j] = kernel_value

    return kernel_matrix


def stable_log_likelihood(X_train, Y_train, M=10, sigma=100, frequencies=[440]):
    Y_train = Y_train.ravel()
    K = MoG_spectral_kernel_matrix(X_train, X_train, M=M, sigma=sigma, frequencies=frequencies) + \
        noise**2 * np.eye(len(X_train))
    L = cholesky(K)

    S1 = solve_triangular(L, Y_train, lower=True)
    S2 = solve_triangular(L.T, S1, lower=False)

    return np.sum(np.log(np.diagonal(L))) + \
        0.5 * Y_train.dot(S2) + \
        0.5 * len(X_train) * np.log(2*np.pi)


# Wav file method
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/'+name+'.wav'

# Read a Wav file
sample_rate, data = wav.read(wav_file)
data = data[:2000]
time_length_of_sample = len(data)/sample_rate
number_of_frames = int(len(data)/1000)
frames = np.array_split(data, number_of_frames)


for frame in frames:
    frame = frame.reshape(-1, 1)
    time_length = len(frame)/sample_rate
    x_train = np.linspace(0, time_length, len(frame)).reshape(-1, 1)
    print("Likelihood: ", stable_log_likelihood(
        x_train, frame, frequencies=[440]))


# frame_size = int(len(data)/1000)
# frame1 = data[:frame_size].reshape(-1, 1)
# time_length_of_frame_size = frame_size/sample_rate
# x_train = np.linspace(0, time_length_of_frame_size, frame_size).reshape(-1, 1)


# print(frame_size)

# print(stable_log_likelihood(x_train, frame1, frequencies=[440]))
