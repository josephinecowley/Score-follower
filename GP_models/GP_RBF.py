from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()


# THINK THIS DOESN"T WORK ANYMORE
# def MoG_spectral_kernel(t1, t2, M=3, sigma=3.0, frequency=261):
#     """
#     Kernel with a Mixture of Gaussian frequency spectrum

#     Args:
#         t1: Array of m points (m x d).
#         t2: Array of n points (n x d).

#     Returns:
#         (m x n) matrix."""

#     omega = 2 * np.pi * frequency
#     sum_of_cosines = 0

#     sqdist = np.sum(t1**2, 1).reshape(-1, 1) + \
#         np.sum(t2**2, 1) - 2 * np.dot(t1, t2.T)
#     for m in range(M):
#         sum_of_cosines += np.cos((m + 1) * omega * (t2 - t1))

#     return 1/(2 * np.pi * sigma**2) * np.exp(-(sigma**2 * sqdist) / 2) * sum_of_cosines



def MoG_spectral_kernel_matrix(X1, X2, M=3, sigma=0.1, frequencies=[440, 880]):
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

def RBF_kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
        np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

X = np.arange(-5, 5, 0.2).reshape(-1, 1)

# Mean and covariance of the prior
mu = np.zeros(X.shape)
cov = MoG_spectral_kernel_matrix(X, X)

# Plot heat map of covariance function
plt.imshow(cov, cmap='hot', interpolation='nearest')
plt.show()


# Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 5)

# Plot GP mean, uncertainty region and samples
plot_gp(mu, cov, X, samples=samples)
plt.show()


def posterior(X_s, X_train, Y_train, M=3, sigma=1.0, sigma_y=1e-8): # for RBF,viola, l = 0.00005 and sigma_f = 2 are optimal
    """
    Computes the sufficient statistics of the posterior distribution 
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    K = MoG_spectral_kernel_matrix(X_train, X_train) + \
        sigma_y**2 * np.eye(len(X_train))
    K_s = MoG_spectral_kernel_matrix(X_train, X_s)
    K_ss = MoG_spectral_kernel_matrix(X_s, X_s) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (7)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


# Option 1: wave file method
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/viola_perfect_5th.wav'
# wav_file = create_sine_wave("Sine.wav", frequency=440)

# Read a WAV file
sample_rate, data = wav.read(wav_file)

Y_train = data[:70].reshape(-1, 1)  # Truncate data to make manageable

# Find time time length of truncated data
time_length = Y_train.shape[0] / sample_rate

# # Plotting the wave form in the time domain
X_train = np.linspace(0., time_length, Y_train.shape[0]).reshape(-1, 1)
X = np.linspace(0, time_length, 200).reshape(-1, 1) 



noise = 0.4
"""
# Option 2: Automatically generated noisy data

X_train = np.arange(-3, 4, 1).reshape(-1, 1)
Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)
X = np.linspace(-5,5,100).reshape(-1,1)
"""

# Compute mean and covariance of the posterior distribution

mu_s, cov_s = posterior(X, X_train, Y_train, sigma_y=noise)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
plt.show()
