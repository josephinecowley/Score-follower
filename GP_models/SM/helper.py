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


# ----------------------------------------------------------
# Helper plotting functions
# ----------------------------------------------------------

def plot_gp(mu, cov, T, T_train=None, Y_train=None, samples=None):
    """
    Plot the Gaussian Process (GP).

    # TODO need to check all the dimension stuff going on

    Args:
        mu: Mean of the GP.
        cov: Covariance matrix of the GP, dimensions (D, D).
        T: Vector of time samples to plot GP for, dimension D.
        T_train: Vector of training time samples.
        Y_train: Vector of training outputs corresponding to T_train.
        samples: Integer number of random samples to draw and plot. 
    """
    T = T.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))  # Plot 95% curves

    plt.fill_between(T, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(T, mu, label='Mean')

    # Generate and plot random samples
    for i in range(samples):
        z = np.random.randn(len(T), 1).ravel()
        # Add a small amount of noise to ensure it is ositive definite
        K = cov + 1e-6 * np.eye(len(T))
        L = np.linalg.cholesky(K)
        y = np.dot(L.T, z)
        plt.plot(T, y, lw=1, ls='--', label=f'Sample {i+1}')

    # Plot training data if required
    if T_train is not None:
        plt.plot(T_train, Y_train, 'rx')

    plt.legend()
    plt.show()

# ----------------------------------------------------------
# Helper kernel functions and matrices
# ----------------------------------------------------------


def SM_kernel(t1, t2, M=6, f=[440], sigma_f=1e-5):
    """
    Spectral Mixture kernel.

    Args:
        t1: Scalar corresponding to time sample One.
        t2: Scalar corresponding to time sample Two.
        M: Integer corresponding to the number of partials.
        f: Array of integers corresponding to different sources.
        sigma_f: Standard deviation of kernel.

    Returns:
        Scalar output of kernel function. 

    TODO add weights k
    TODO add variance changes across Qs and Ms
    """
    cosine_series = 0
    for fundamental_frequency in f:
        for m in range(M):
            cosine_series += (1/(1+((1/fundamental_frequency)*(m+1))**v)) * np.cos((m+1) * 2 * np.pi *
                                                                                   fundamental_frequency * np.linalg.norm(t1 - t2))
    return np.exp(-(sigma_f**2) * 2 * np.pi**2 * np.linalg.norm(t1 - t2)**2) * cosine_series


def SM_kernel_matrix(T1, T2=None, M=6, f=[440], sigma_f=1e-5):
    """
    Compute the Spectral Mixture kernel matrix between two sets of vectors.
    If T2 is None, find matrix between T1 and T1.

    Args:
        T1: Array of time samples of shape (n1, d) where n1 is the number of vectors and d is the dimension of each vector.
        T2: Array of time samples of shape (n2, d) where n2 is the number of vectors and d is the dimension of each vector.

    Returns:
        Matrix of SM kernel.
    """
    if T2 is None:
        T2 = T1

    n1 = T1.shape[0]
    n2 = T2.shape[0]

    kernel_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            t1 = T1[i, :]
            t2 = T2[j, :]

            kernel_matrix[i, j] = SM_kernel(t1, t2, M, sigma_f, f)

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

# ----------------------------------------------------------
# Marginal likelihood functions
# ----------------------------------------------------------


def nlml(T, Y, M=10, sigma_f=100, frequencies=[400], noise=1e-5):
    """
    Return Negative Log Marginal Likelihood.
    Assumes zero mean, and does not add noise to ensure positive definite covariance matrix.

    Args:
        T: Vector of input time samples.
        Y: Vector of corresponding time value outputs, to be measured against the GP model.

    Returns:
        Value of Negative Log Marginal Likelihood.
    """
    Y = Y.ravel()
    K = SM_kernel_matrix(T, T, M=M, sigma_f=sigma_f, frequencies=frequencies) + \
        noise**2 * np.eye(len(T))
    return 0.5 * Y.dot(inv(K).dot(Y)) + 0.5 * np.log(det(K)) + 0.5 * len(T) * np.log(2*np.pi)


def stable_nlml(T, Y, M=6, sigma_f=1e-5, f=[440], sigma_n=1e-5):
    """
    Return Negative Log Marginal Likelihood via stable method.
    Assumes zero mean.

    Numerically more stable implementation of nlml as described in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, 
    Section 2.2, Algorithm 2.1.

    Args:
        T: Vector of input time samples.
        Y: Vector of corresponding time value outputs, to be measured against the GP model.

    Returns:
        Value of Negative Log Marginal Likelihood.
    """
    Y = Y.ravel()
    K = SM_kernel_matrix(T, T, M=M, sigma_f=sigma_f, f=f) + \
        sigma_n**2 * np.eye(len(T))
    L = cholesky(K)

    S1 = solve_triangular(L, Y, lower=True)
    S2 = solve_triangular(L.T, S1, lower=False)

    return np.sum(np.log(np.diagonal(L))) + \
        0.5 * Y.dot(S2) + \
        0.5 * len(T) * np.log(2*np.pi)

# ----------------------------------------------------------
# Predictor functions
# ----------------------------------------------------------


def posterior(X_s, X_train, Y_train, M=10, sigma=1.0, sigma_y=1e-8, frequencies=[440, 880]):
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
    K = SM_kernel_matrix(X_train, X_train, M=M, sigma=sigma, frequencies=frequencies) + \
        sigma_y**2 * np.eye(len(X_train))
    K_s = SM_kernel_matrix(
        X_train, X_s, M=M, sigma=sigma, frequencies=frequencies)
    K_ss = SM_kernel_matrix(
        X_s, X_s, M=M, sigma=sigma, frequencies=frequencies) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Calculate posterior mean function
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Calculate posterior covariance function
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s

# ----------------------------------------------------------
# Helper optimiser functions
# ----------------------------------------------------------


def golden_section(x1, x2, X_train, Y_train, M, sigma, tol=1, integer_search=False):
    # Initial points
    f1 = stable_nlml(X_train, Y_train, M, sigma, frequencies=[x1])
    f2 = stable_nlml(X_train, Y_train, M, sigma, frequencies=[x2])

    # Set up golden ratios
    r = (np.sqrt(5)-1)/2.0

    # Third point
    if integer_search is False:
        x3 = x1 * (1-r) + x2 * r
    else:
        x3 = int(x1 * (1-r) + x2 * r)
    f3 = stable_nlml(X_train, Y_train, M, sigma, frequencies=[x3])

    # Loop until convergence
    while abs(x1-x2) > tol:

        if integer_search is False:
            x4 = x1 * r + x2 * (1-r)
        else:
            x4 = int(x1 * r + x2 * (1-r))
        f4 = stable_nlml(
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


def nll_fn(X_train, Y_train, sigma_n, naive=True):
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

    Example:
    res = minimize(nll_fn(X_train, Y_train, noise), [1, 1],
               bounds=((1e-5, None), (1e-5, None)),
               method='L-BFGS-B')
    l_opt, sigma_f_opt = res.x

    """

    Y_train = Y_train.ravel()

    def nlml_naive(theta):
        # Naive implementation of Eq. (11). Works well for the examples
        # in this article but is numerically less stable compared to
        # the implementation in nll_stable below.
        K = SM_kernel_matrix(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            sigma_n**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
            0.5 * Y_train.dot(inv(K).dot(Y_train)) + \
            0.5 * len(X_train) * np.log(2*np.pi)

    def nlml_stable(theta):
        # Numerically more stable implementation of Eq. (11) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.

        K = SM_kernel_matrix(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            sigma_n**2 * np.eye(len(X_train))
        L = cholesky(K)

        S1 = solve_triangular(L, Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)

        return np.sum(np.log(np.diagonal(L))) + \
            0.5 * Y_train.dot(S2) + \
            0.5 * len(X_train) * np.log(2*np.pi)

    if naive:
        return nlml_naive
    else:
        return nlml_stable
