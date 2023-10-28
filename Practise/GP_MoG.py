from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from numpy.linalg import cholesky, det


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


def MoG_spectral_kernel(x1, x2, M=3,  sigma=0.00001, frequencies=[440, 500]):
    """MoG spectral kernel
    M is the number of partials or harmonics for each note source
    TODO add weights k
    TODO add variance changes across Qs and Ms
    """
    cosine_series = 0
    for fundamental_frequency in frequencies:
        for m in range(M):
            cosine_series += np.cos((m+1) * 2 * np.pi *
                                    fundamental_frequency * np.linalg.norm(x1 - x2))
    return np.exp(-(sigma**2) * 2*np.pi**2 * np.linalg.norm(x1 - x2)**2) * cosine_series


def plot_cov_matrix(X, Y=None, M=1,  sigma=100, frequencies=[440, 880]):
    if Y is None:
        Y = X
    cov = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            cov[i, j] = MoG_spectral_kernel(
                i, j, M=M, sigma=sigma, frequencies=frequencies)
    return cov


# Plot the kernel function
X = np.arange(-50, 50, 1)
data1 = np.zeros(len(X))
data2 = np.zeros(len(X))
data3 = np.zeros(len(X))
for i in range(len(X)):
    data1[i] = MoG_spectral_kernel(X[i], 0, sigma=0.1)
    data2[i] = MoG_spectral_kernel(X[i], 0, sigma=0.00001)
    data3[i] = MoG_spectral_kernel(

        X[i], 0, sigma=0.1, frequencies=[440, 880, 0.05])
fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(X, data1)
axes[0, 0].set_title("sigma = 0.1")
axes[0, 1].plot(X, data2)
axes[0, 1].set_title("sigma = 0.00001")
axes[1, 0].plot(X, data3)
axes[1, 0].set_title("frequencies=[440,880]")
fig.tight_layout()
fig.show()


# Mean of the prior
X = np.linspace(-5, 5, 100)
cov = plot_cov_matrix(X, X)
mu = np.zeros(X.shape)

# Plot heat map of covariance function
plt.imshow(cov, cmap='hot', interpolation='nearest')
plt.show()


# # Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 5)

# # Plot GP mean, uncertainty region and samples
plot_gp(mu, cov, X, samples=samples)
plt.show()


def posterior(X_s, X_train, Y_train, M=1, sigma=0.001,  sigma_y=1e-8):
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
    K = plot_cov_matrix(X_train, X_train, M=M, sigma=sigma) + \
        sigma_y**2 * np.eye(len(X_train))
    K_s = plot_cov_matrix(X_train, X_s, M=M, sigma=sigma)
    K_ss = plot_cov_matrix(X_s, X_s, M=M, sigma=sigma) + \
        1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (7)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


# Option 1: wave file method
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/tuner_440.wav'
# wav_file = create_sine_wave("Sine.wav", frequency=440)

# # Read a WAV file
sample_rate, data = wav.read(wav_file)

Y_train = data[:200].reshape(-1, 1)  # Truncate data to make manageable

# # Find time time length of truncated data
time_length = Y_train.shape[0] / sample_rate

# # Plotting the wave form in the time domain
# Had to change time_length to 10 inorder to see changes in kernel function
X_train = np.linspace(0., time_length, Y_train.shape[0]).reshape(-1, 1)
# For some reason I need to make this the same shape as the training data - need to look into this
X = np.linspace(0, 2 * time_length, 400).reshape(-1, 1)

noise = 0.004

# # Option 2: Automatically generated noisy data

# # X_train = np.linspace(-5, 5, 50).reshape(-1, 1)
# # Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)
# # X = np.linspace(-5,5,60).reshape(-1,1)


# # Compute mean and covariance of the posterior distribution

mu_s, cov_s = posterior(X, X_train, Y_train, M=3, sigma=0.01, sigma_y=noise)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
plt.show()


# def nll_fn(X_train, Y_train, noise, naive=True):
#     """
#     Returns a function that computes the negative log marginal
#     likelihood for training data X_train and Y_train and given
#     noise level.

#     Args:
#         X_train: training locations (m x d).
#         Y_train: training targets (m x 1).
#         noise: known noise level of Y_train.
#         naive: if True use a naive implementation of Eq. (11), if
#                False use a numerically more stable implementation.

#     Returns:
#         Minimization objective.
#     """

#     Y_train = Y_train.ravel()

#     def nll_naive(theta):
#         # Naive implementation of Eq. (11). Works well for the examples
#         # in this article but is numerically less stable compared to
#         # the implementation in nll_stable below.
#         K = plot_cov_matrix(X_train, X_train, M=int(theta[0]), sigma=theta[1]) + \
#             noise**2 * np.eye(len(X_train))
#         return 0.5 * np.log(det(K)) + \
#             0.5 * Y_train.dot(inv(K).dot(Y_train)) + \
#             0.5 * len(X_train) * np.log(2*np.pi)

#     def nll_stable(theta):
#         # Numerically more stable implementation of Eq. (11) as described
#         # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
#         # 2.2, Algorithm 2.1.

#         K = plot_cov_matrix(X_train, X_train, M=int(theta[0]), sigma=theta[1]) + \
#             noise**2 * np.eye(len(X_train))
#         L = cholesky(K)

#         S1 = solve_triangular(L, Y_train, lower=True)
#         S2 = solve_triangular(L.T, S1, lower=False)

#         return np.sum(np.log(np.diagonal(L))) + \
#             0.5 * Y_train.dot(S2) + \
#             0.5 * len(X_train) * np.log(2*np.pi)

#     if naive:
#         return nll_naive
#     else:
#         return nll_stable


# # Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.
# # We should actually run the minimization several times with different
# # initializations to avoid local minima but this is skipped here for
# # simplicity.
# res = minimize(nll_fn(X_train, Y_train, noise), [1, 0.1],
#                bounds=((1e-5, None), (1e-5, None)),
#                method='L-BFGS-B')

# # Store the optimization results in global variables so that we can
# # compare it later with the results from other implementations.
# m, sigma_opt = res.x
# print(m, sigma_opt)
# # Compute posterior mean and covariance with optimized kernel parameters and plot the results
# mu_s, cov_s = posterior(X, X_train, Y_train, M=int(m),
#                         sigma=sigma_opt, sigma_y=noise)
# plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
# plt.show()
