from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.linalg import inv
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

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


def MoG_spectral_kernel_matrix(X1, X2, M=20, sigma=0.1, frequencies=[4, 5]):
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


def posterior(X_s, X_train, Y_train, M=20, sigma=1.0, sigma_y=1e-8, frequencies = [440, 880]): # for RBF,viola, l = 0.00005 and sigma_f = 2 are optimal
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


# Option 1: wave file method
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/tuner_440.wav'
# wav_file = create_sine_wave("Sine.wav", frequency=440)

# Read a WAV file
sample_rate, data = wav.read(wav_file)

Y_train = data[:200].reshape(-1, 1)  # Truncate data to make manageable

# Find time length of truncated data
time_length = Y_train.shape[0] / sample_rate

# # Plotting the wave form in the time domain
X_train = np.linspace(0., time_length, Y_train.shape[0]).reshape(-1, 1)
X = np.linspace(0, time_length, 200).reshape(-1, 1) 

noise = 0.0005
"""
# Option 2: Automatically generated noisy data

X_train = np.arange(-3, 4, 1).reshape(-1, 1)
Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)
X = np.linspace(-5,5,100).reshape(-1,1)
"""

# Compute mean and covariance of the posterior distribution
mu_s, cov_s = posterior(X, X_train, Y_train, sigma_y=noise)

# samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
plt.show()


def nll_fn(X_train, Y_train, noise, M=20,  naive=True):
    """
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        naive: if True use a naive implementation of Eq. (11), if
               False use a numerically more stable implementation.

    Returns:
        Minimization objective.
    """
    
    Y_train = Y_train.ravel()
    
    def nll_naive(theta):
        # Naive implementation of Eq. (11). Works well for the examples 
        # in this article but is numerically less stable compared to 
        # the implementation in nll_stable below.
        K = MoG_spectral_kernel_matrix(X_train, X_train, M=20, sigma=theta[0], frequencies = [theta[1]]) + \
            noise**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
               0.5 * Y_train.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2*np.pi)
        
    def nll_stable(theta):
        # Numerically more stable implementation of Eq. (11) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        
        K = MoG_spectral_kernel_matrix(X_train, X_train, M=20, sigma=theta[0], frequencies = [theta[1]]) + \
            noise**2 * np.eye(len(X_train))
        L = cholesky(K)
        
        S1 = solve_triangular(L, Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)
        
        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * Y_train.dot(S2) + \
               0.5 * len(X_train) * np.log(2*np.pi)

    if naive:
        return nll_naive
    else:
        return nll_stable
    
    
# Minimize the negative log-likelihood w.r.t. parameters m and sigma, and the first two frequencies.
# (JC: We should actually run the minimization several times with different
# initializations to avoid local minima but this is skipped here for
# simplicity.)
res = minimize(nll_fn(X_train, Y_train,noise,  M=20), [ 1, 450 ], 
               bounds=((1e-5, 1e4), (20,20000)),
               method='L-BFGS-B')

# Store the optimization results in global variables so that we can
# compare it later with the results from other implementations.
sigma_f_opt, frequencies_1 = res.x
print( sigma_f_opt, [frequencies_1])
# # # Compute posterior mean and covariance with optimized kernel parameters and plot the results
mu_s, cov_s = posterior(X, X_train, Y_train, M=20, sigma=sigma_f_opt, sigma_y=noise, frequencies=[frequencies_1])
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)

# Now want to calculate optimal value for M
plt.show()
