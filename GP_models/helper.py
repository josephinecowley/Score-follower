from __future__ import print_function

import numpy as np
from numpy.linalg import inv
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from . import inharmonicity


def SM_kernel(X1, X2, f: list = [440], M: int = 9, sigma_f: float = 0.005, w: list = None, T: float = 0.465, v: float = 2.37, B: dict = None):
    """
    SM kernel. Takes two arrays and returns a matrix with elements of the SM covariance function where K_{i,j} = k(x_i,x_j).

    Args:
        X1: Array of m points.
        X2: Array of n points.

        f: List of frequencies GP hyperparameter. 
        M: Integer for number of harmonics (Including the fundamental frequency).
        sigma_f: Float for inverse length scale GP hyperparameter.
        w: List of relative weights GP hyperparameter.
        T: Float for parameter of spectral envelope weights (E_m) GP hyperparameter.
        v: Float for parameter of spectral envelope weights (E_m) GP hyperparameter.
        B: Dictionary of inharmonicity constants GP hyperparameter.

    Returns:
        Covariance matrix (K).
    """
    M = int(M)

    if v is None:
        v = 2.37
    if T is None:
        T = 0.465
    if w is None:
        w = np.ones(len(f))/np.sum(len(f))

    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)

    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
        np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

    cosine_series = np.zeros((X1.shape[0], X2.shape[0]))

    for i, fundamental_frequency in enumerate(f):
        if B is None:
            closest_key = min(inharmonicity.B.keys(), key=lambda key: abs(
                key - fundamental_frequency))
            B = inharmonicity.B[closest_key]
        for m in range(M):
            inharmonicity_const = np.sqrt((1 + B * (m+1)**2))
            k_m = 2 * np.pi * inharmonicity_const * \
                (m+1) * fundamental_frequency
            A = k_m * X1
            C = k_m * X2
            cosine_series += w[i] / (1 + (T*(m+1))**v) * (
                np.cos(A) * np.cos(C).T + np.sin(A) * np.sin(C).T)

    return np.exp(-2 * np.pi**2 * sigma_f**2 * sqdist) * cosine_series


def nlml(time_samples, Y, cov_s=None, M=10, sigma_f=1/500000, f=[400],  B=None, sigma_n=1e-2, T=None, v=None, amplitude=None):
    """
    Return Negative Log Marginal Likelihood.
    Assumes zero mean, and does not add noise to ensure positive definite covariance matrix.

    Args:
        time_samples: Vector of input time samples.
        Y: Vector of corresponding time value outputs, to be measured against the GP model.

    Returns:
        Value of Negative Log Marginal Likelihood.
    """
    Y = Y.ravel()
    if cov_s is None:
        K = SM_kernel(time_samples, time_samples, M=M, sigma_f=sigma_f, f=f, B=B, T=T, v=v, w=amplitude) + \
            sigma_n**2 * np.eye(len(time_samples))
    else:
        K = cov_s + sigma_n**2 * np.eye(len(time_samples))
    return 0.5 * Y.dot(inv(K).dot(Y)) + 0.5 * np.log(det(K)) + 0.5 * len(time_samples) * np.log(2*np.pi)


def stable_nlml(x, y, f: list = [440], M: int = 9, sigma_f: float = 0.005, w: list = None, T: float = 0.465, v: float = 2.37, B: dict = None, sigma_n: float = 1e-2, cov_dict: dict = None):
    """
    Return negative log marginal likelihood via stable efficient method (using Cholesky factorisation).
    Assumes zero mean.

    Args:
        y: Array of input audioframe values.
        x: Array of input audio sample times.

        f: List of frequencies GP hyperparameter. 
        M: Integer for number of harmonics (Including the fundamental frequency).
        sigma_f: Float for inverse length scale GP hyperparameter.
        w: List of relative weights GP hyperparameter.
        T: Float for parameter of spectral envelope weights (E_m) GP hyperparameter.
        v: Float for parameter of spectral envelope weights (E_m) GP hyperparameter.
        B: Dictionary of inharmonicity constants GP hyperparameter.
        sigma_n: Float for additive Gaussian noise GP hyperparameter.

        cov_dict: Dictionary of pre-calculated covariance matrices for states in a piece.

    Returns:
        Value of negative log marginal likelihood.
    """
    # if len(Y[0]) != 1:
    #     # We have a multi channel input— choose first channel (arbitrary)
    #     Y = Y[:, 0]
    y = y.ravel()

    if cov_dict is not None and str(f) in cov_dict:
        K = cov_dict[str(f)]  # Note this cov_dict already has noise added
    else:
        K = SM_kernel(x, x, M=M, sigma_f=sigma_f, f=f, B=B, T=T, v=v, w=w) + \
            sigma_n**2 * np.eye(len(x))

    L = cholesky(K)

    S1 = solve_triangular(L, y, lower=True)
    S2 = solve_triangular(L.T, S1, lower=False)

    return np.sum(np.log(np.diagonal(L))) + \
        0.5 * y.dot(S2) + \
        0.5 * len(x) * np.log(2*np.pi)


def posterior(T_test, T_train, Y_train, M=14, sigma_f=1/500000, sigma_y=0.0001, f=[440], B=None, T=None, v=None, amplitude=None):
    """
    Computes the sufficient statistics of the posterior distribution 
    from m training data T_train and Y_train and n new inputs T_test.

    Args:
        T_test: New input locations (n x d).
        T_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        M: Number of partials.
        sigma: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    T_train = T_train.reshape(-1, 1)
    T_test = T_test.reshape(-1, 1)
    K = SM_kernel(T_train, T_train, M=M, sigma_f=sigma_f, f=f, B=B, T=T, v=v, w=amplitude) + \
        sigma_y**2 * np.eye(len(T_train))
    K_s = SM_kernel(
        T_train, T_test, M=M, sigma_f=sigma_f, f=f, B=B, T=T, v=v, w=amplitude)
    K_ss = SM_kernel(
        T_test, T_test, M=M, sigma_f=sigma_f, f=f, B=B, T=T, v=v, w=amplitude) + 1e-8 * np.eye(len(T_test))
    K_inv = inv(K)

    # Calculate posterior mean function
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Calculate posterior covariance function
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s
