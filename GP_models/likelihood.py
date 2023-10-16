from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.linalg import inv
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.fft import fft
from scipy.fft import fftfreq

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov)) # Plot 95% curves

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()


def MoG_spectral_kernel_matrix(X1, X2, M=10, sigma=100000, frequencies=[600, 1200]):
    """
    Compute the MoG spectral kernel matrix between two sets of vectors.
    
    X1 and X2 are arrays of shape (n1, d) and (n2, d), where n1 and n2 are the
    numbers of vectors and d is the dimension of each vector.
    
    M is the number of partials or harmonics for each note source.
    """
    n1= X1.shape[0]
    n2 = X2.shape[0]
    
    kernel_matrix = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            x1 = X1[i, :]
            x2 = X2[j, :]
            
            cosine_series = 0
            for fundamental_frequency in frequencies:
                for m in range(M):
                    cosine_series += np.cos((m + 1) * 2 * np.pi * fundamental_frequency * np.linalg.norm(x1 - x2))
            
            kernel_value = (1 / np.pi) * np.exp(-(sigma**2 / 2) * np.linalg.norm(x1 - x2)**2) * cosine_series
            kernel_matrix[i, j] = kernel_value
    
    return kernel_matrix

def posterior(X_s, X_train, Y_train, M=10, sigma=1.0, sigma_y=1e-8, frequencies = [440]): # for RBF,viola, l = 0.00005 and sigma_f = 2 are optimal
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
    K_s = MoG_spectral_kernel_matrix(X_train, X_s, M=M, sigma=sigma, frequencies=frequencies)
    K_ss = MoG_spectral_kernel_matrix(X_s, X_s, M=M, sigma=sigma, frequencies=frequencies) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Calculate posterior mean function
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Calculate posterior covariance function
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


# Wav file method
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/viola_octave.wav'

# Read a Wav file
sample_rate, data = wav.read(wav_file)

# Truncate data to make manageable
Y_train = data[:100].reshape(-1, 1)  

# Find time length of truncated data
time_length = Y_train.shape[0] / sample_rate

# # Plotting the wave form in the time domain
X_train = np.linspace(0., time_length, Y_train.shape[0]).reshape(-1, 1)
X = np.linspace(0, time_length, 200).reshape(-1, 1) 

noise = 0.0005

# Compute mean and covariance of the posterior distribution
mu_s, cov_s = posterior(X, X_train, Y_train, M = 20,sigma_y=noise)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
plt.show()

# # Convert audio data to frequency domain
# fft_data = fft(mu_s)
# N = len(mu_s)
# normalise = N/2

# # Get the frequency components of the spectrum
# frequency_axis = fftfreq(N, d=1/44100)
# norm_amplitude = np.abs(fft_data)/normalise

# # Plot the results
# plt.plot(frequency_axis, norm_amplitude)
# plt.xlabel('Frequency[Hz]')
# plt.ylabel('Amplitude')
# plt.title('Spectrum')
# plt.show()

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

print(f"loglikelihood of 586 Hz is {stable_log_likelihood(X_train=X_train, Y_train=Y_train, frequencies=[586])}")
print(f"loglikelihood of 440 Hz is {stable_log_likelihood(X_train=X_train, Y_train=Y_train, frequencies=[440])}")
print(f"loglikelihood of 5 Hz is {stable_log_likelihood(X_train=X_train, Y_train=Y_train, frequencies=[5])}")

res = minimize(stable_log_likelihood(X_train, Y_train, noise),  [frequencies=[440]] , 
               bounds=((20,20000)),
               method='L-BFGS-B')