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


def MoG_spectral_kernel(x1, x2, M=3,  sigma=0.1, frequencies=[440]):
    """MoG spectral kernel
    M is the number of partials or harmonics for each note source
    TODO add weights k
    TODO add variance changes across Qs and Ms
    """
    cosine_series = 0
    for fundamental_frequency in frequencies:    
        for m in range(M):
            cosine_series += np.cos((m+1) * 2 * np.pi * fundamental_frequency * np.linalg.norm(x1 - x2))
    return (1/np.pi ) * np.exp(-(sigma**2/2) * np.linalg.norm(x1- x2)**2) * cosine_series


def plot_cov_matrix(X, Y=None):
    if Y is None:
        Y = X
    cov = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            cov[i,j] = MoG_spectral_kernel(i, j)
    return cov


# Mean of the prior
X = np.linspace(-5,5,100)
cov = plot_cov_matrix(X, X)
mu = np.zeros(X.shape)

# Plot heat map of covariance function
plt.imshow(cov, cmap='hot', interpolation='nearest')
plt.show()


# Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 5)

# # Plot GP mean, uncertainty region and samples
plot_gp(mu, cov, X, samples=samples)
plt.show()


def posterior(X_s, X_train, Y_train,  M=10, sigma=5., frequency=265, sigma_y=1e-8): 
    """
    Computes the suffifient statistics of the posterior distribution 
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
    K = plot_cov_matrix(X_train, X_train) + \
        sigma_y**2 * np.eye(len(X_train))
    K_s = plot_cov_matrix(X_train, X_s)
    K_ss = plot_cov_matrix(X_s, X_s) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (7)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


# Option 1: wave file method
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/Sine.wav'
# wav_file = create_sine_wave("Sine.wav", frequency=440)

# # Read a WAV file
sample_rate, data = wav.read(wav_file)

Y_train = data[:200].reshape(-1, 1)  # Truncate data to make manageable

# # Find time time length of truncated data
time_length = Y_train.shape[0] / sample_rate

# # Plotting the wave form in the time domain
X_train = np.linspace(0., 10, Y_train.shape[0]).reshape(-1, 1) # Had to change time_length to 10 inorder to see changes in kernel function
X = np.linspace(0, 10, 200).reshape(-1, 1) # For some reason I need to make this the same shape as the training data - need to look into this




noise = 0.4

# Option 2: Automatically generated noisy data

# X_train = np.linspace(-5, 5, 50).reshape(-1, 1)
# Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)
# X = np.linspace(-5,5,60).reshape(-1,1)


# Compute mean and covariance of the posterior distribution

mu_s, cov_s = posterior(X, X_train, Y_train, sigma = 3, sigma_y=noise, M=20, frequency = 4400)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
plt.show()
