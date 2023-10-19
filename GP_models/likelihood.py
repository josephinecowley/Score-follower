from __future__ import print_function

from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.linalg import inv
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import golden
from scipy.fft import fft
from scipy.fft import fftfreq
import scipy.io.wavfile as wavf

name = 'tuner_440'


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))  # Plot 95% curves

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()


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


# for RBF,viola, l = 0.00005 and sigma_f = 2 are optimal
def posterior(X_s, X_train, Y_train, M=10, sigma=1.0, sigma_y=1e-8, frequencies=[440]):
    """
    Computes the sufficient statistics of the posterior distribution 
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        M: Number of partials.
        sigma: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    K = MoG_spectral_kernel_matrix(X_train, X_train, M=M, sigma=sigma, frequencies=frequencies) + \
        sigma_y**2 * np.eye(len(X_train))
    K_s = MoG_spectral_kernel_matrix(
        X_train, X_s, M=M, sigma=sigma, frequencies=frequencies)
    K_ss = MoG_spectral_kernel_matrix(
        X_s, X_s, M=M, sigma=sigma, frequencies=frequencies) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Calculate posterior mean function
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Calculate posterior covariance function
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s

# Convert data to frequency domain and plot


def plot_frequency_response(data, data2, sample_rate):
    data = data.ravel()
    fft_data = fft(data)
    N = len(data)
    normalise = N/2

    frequency_axis = fftfreq(N, 1.0/sample_rate)
    norm_amplitude = np.abs(fft_data)/normalise

    plt.plot(frequency_axis[:N//2], norm_amplitude[:N//2], 'b')
    if data2 is not None:
        data2 = data2.ravel()
        fft_data2 = fft(data2)
        N2 = len(data2)
        normalise2 = N2/2
        norm_amplitude_2 = np.abs(fft_data2)/normalise2
        plt.plot(frequency_axis[:N//2], norm_amplitude_2[:N//2], 'r')
    # Plot the results
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplitude')
    plt.title('Spectrum')
    plt.legend(["posterior", "actual data"])
    plt.show()


# Wav file method
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/'+name+'.wav'

# Read a Wav file
sample_rate, data = wav.read(wav_file)

# Truncate data to make manageable
Y_train = data[:200].reshape(-1, 1)

# Find time length of truncated data
time_length = Y_train.shape[0] / sample_rate

# Plotting the wave form in the time domain
X_train = np.linspace(0., time_length, Y_train.shape[0]).reshape(-1, 1)
X = np.linspace(0, time_length, 200).reshape(-1, 1)

noise = 0.0005

mu_s, cov_s = posterior(X, X_train, Y_train, M=30,
                        sigma_y=noise, frequencies=[440])
plot_frequency_response(mu_s, Y_train, sample_rate)
# # samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
# plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
# plt.show()


# Plot covariance function - prior
zeros = np.zeros(X_train.shape)
cov = MoG_spectral_kernel_matrix(X_train, X_train)
# Plot covariance function - posterior
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("prior covariance function")
ax2.set_title("posterior covariance function")
ax1.imshow(cov, cmap='hot', interpolation='nearest')
ax2.imshow(cov_s, cmap='hot', interpolation='nearest')
plt.show()


# Plot posterior function over points
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
plt.show()


# Play modelled sound
data = np.array([mu_s]*10)
out_f = 'out.wav'
wavf.write(out_f, sample_rate, data/24500)


def log_likelihood(X_train, Y_train, M=10, sigma=100, frequencies=[400]):
    Y_train = Y_train.ravel()
    K = MoG_spectral_kernel_matrix(X_train, X_train, M=M, sigma=sigma, frequencies=frequencies) + \
        noise**2 * np.eye(len(X_train))
    print(det(K))
    return 0.5 + 0.5 * Y_train.dot(inv(K).dot(Y_train)) + 0.5 * len(X_train) * np.log(2*np.pi)


def stable_log_likelihood(X_train, Y_train, M=10, sigma=100, frequencies=[400]):
    Y_train = Y_train.ravel()
    K = MoG_spectral_kernel_matrix(X_train, X_train, M=M, sigma=sigma, frequencies=frequencies) + \
        noise**2 * np.eye(len(X_train))
    L = cholesky(K)

    S1 = solve_triangular(L, Y_train, lower=True)
    S2 = solve_triangular(L.T, S1, lower=False)

    return np.sum(np.log(np.diagonal(L))) + \
        0.5 * Y_train.dot(S2) + \
        0.5 * len(X_train) * np.log(2*np.pi)


def golden_section(x1, x2, X_train, Y_train, M, sigma, tol=1):
    # Initial points
    f1 = stable_log_likelihood(X_train, Y_train, M, sigma, frequencies=[x1])
    f2 = stable_log_likelihood(X_train, Y_train, M, sigma, frequencies=[x2])

    # Set up golden ratios
    r = (np.sqrt(5)-1)/2.0

    # Third point
    x3 = x1 * (1-r) + x2 * r
    f3 = stable_log_likelihood(X_train, Y_train, M, sigma, frequencies=[x3])

    # Loop until convergence
    while abs(x1-x2) > tol:

        x4 = x1 * r + x2 * (1-r)
        f4 = stable_log_likelihood(
            X_train, Y_train, M, sigma, frequencies=[x4])
        print(f4, '/n', x3)
        if f4 < f3:
            x2 = x3
            f2 = f3
            x3 = x4
            f3 = f4
        else:
            x1 = x2
            f1 = f2
            x2 = x4
            f2 = f4
    return x3


def m_golden_section(x1, x2, X_train, Y_train,  sigma, frequencies, tol=1):
    """
    Computes the optimal value for M, the number of partials 
    """
    # Initial points
    f1 = stable_log_likelihood(X_train, Y_train, x1, sigma, frequencies)
    f2 = stable_log_likelihood(X_train, Y_train, x2, sigma, frequencies)

    # Set up golden ratios
    r = (np.sqrt(5)-1)/2.0

    # Third point
    x3 = int(x1 * (1-r) + x2 * r)
    f3 = stable_log_likelihood(X_train, Y_train, x3, sigma, frequencies)

    # Loop until convergence
    while abs(x1-x2) > tol:

        x4 = int(x1 * r + x2 * (1-r))
        f4 = stable_log_likelihood(X_train, Y_train, x4, sigma, frequencies)
        print(f4, x3)
        if f4 < f3:
            x2 = x3
            f2 = f3
            x3 = x4
            f3 = f4
        else:
            x1 = x2
            f1 = f2
            x2 = x4
            f2 = f4
    return x3

# M_opt = m_golden_section(1, 50, X_train, Y_train, sigma=10, frequencies=[1320])
# print(M_opt)


# frequency_opt = golden_section( 300, 1000, X_train, Y_train, 14, 1)
# print(frequency_opt)


# # Compute mean and covariance of the posterior distribution with optimal m
# mu_s, cov_s = posterior(X, X_train, Y_train, M = M_opt, sigma_y=noise, frequencies=[frequency_opt])
# plot_frequency_response(mu_s, Y_train, sample_rate)
# # samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
# plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
# plt.show()


# x = [45, 400,440, 500, 600,  800, 1300]
# likelihoods = [stable_log_likelihood(X_train, Y_train,frequencies=[a]) for a in x]
# print(likelihoods)
# plt.scatter(x,likelihoods )
# plt.show()

# print(f"loglikelihood of 586 Hz is {stable_log_likelihood(X_train=X_train, Y_train=Y_train, frequencies=[586])}")
# print(f"loglikelihood of 440 Hz is {stable_log_likelihood(X_train=X_train, Y_train=Y_train, frequencies=[440])}")
# print(f"loglikelihood of 5 Hz is {stable_log_likelihood(X_train=X_train, Y_train=Y_train, frequencies=[5])}")

# res = minimize(return_fn(X_train, Y_train, M=10), [0.01, 420], bounds=[(0.0000005, 100), (20, 20000)], method='L-BFGS-B')
# sigma, frequency=res.x
# print(sigma, frequency)
