from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.linalg import inv
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import golden
from scipy.fft import fft
from scipy.fft import fftfreq
from scipy.signal.windows import hann, hamming
import scipy.io.wavfile as wavf
from tqdm import tqdm
from . import inharmonicity

# ----------------------------------------------------------
#  GP experimental functions
# ----------------------------------------------------------


def plot_audio(T, data, show=False, title="Plot of audio wave"):
    """
    Visualise audio data. 

    Args:
        T: 1D numpy array of time samples.
        data: 1D numpy array of corresponding audio amplitudes.

    Returns:
        Nothing. Simply plots a grah of the audio sample. 
    """
    plt.plot(T, data, 'r')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    if show is True:
        plt.show()


def plot_fft(data, sample_rate=44100, power_spectrum=False, title='Spectrum', colour='r'):
    """
    Visualise audio frequency data using DFT.

    Args: 
        power_spectrum: Make this true if plotting a kernel function as the fft will be the power spectrum.
    Returns:
        Returns positive fft data and plots a graph of positive values of the audio spectrum. 
    """
    w = hann(len(data))
    if power_spectrum is False:
        fft_data = np.abs(fft(data*w, norm="ortho"))
    else:
        fft_data = np.abs(fft(data*w, norm="ortho"))**2
    frequency_axis = fftfreq(len(data), d=1.0/sample_rate)
    plt.plot(frequency_axis[:(len(data)//8)],
             fft_data[:(len(data)//8)], colour, label='Audio')
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplitude')
    plt.title(title)


def psd(audio_samples, sample_rate):
    w = hann(len(audio_samples))
    psd = np.abs(fft(audio_samples*w))**2 / len(audio_samples)
    frequency_axis = fftfreq(len(audio_samples), d=1.0/sample_rate)
    plt.plot(frequency_axis[:(len(audio_samples)//8)],
             psd[:(len(audio_samples)//8)], label='Audio', color='red')
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplitude')
    return psd[:(len(audio_samples)//8)], frequency_axis[:(len(audio_samples)//8)]


def plot_kernel(T, kernel, title="Spectrum of Kernel"):
    """
    Visualise kernel function.

    Args:
        T: Time samples which have been inputs to the kernel function.
        kernel: Corresponding kernel outputs. 

    Returns:
        Plot of kerne function. 
    """
    plt.plot(T, kernel)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Kernel value")
    plt.title(title)
    plt.show()


def plot_kernel_matrix(kernel_matrix, title="Covariance Matrix Heatmap"):
    """
    Visualise kernel matrix.

    Args:
        kernel_matrix: a matrix.

    Returns:
        Plot of matrix in heat map. 
    """
    plt.imshow(kernel_matrix, cmap='coolwarm', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_gp(mu, cov, T_test, T_train=None, Y_train=None, samples=None, title="GP"):
    """
    Plot the Gaussian Process (GP).

    # TODO need to check all the dimension stuff going on

    Args:
        mu: Mean vector of the GP at the test inputs.
        cov: Covariance matrix of the GP, from test inputs.
        T_test: Vector of time samples to plot GP for.
        T_train: Vector of training time samples.
        Y_train: Vector of training outputs corresponding to T_train.
        samples: Integer number of random samples to draw and plot. 

    Returns:
        Plot of predictive mean and 95% error bars, along with the number of samples specified.
    """
    T_test = T_test.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))  # Plot 95% curves

    plt.fill_between(T_test, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(T_test, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(T_test, sample, lw=1, ls='--', label=f'Sample {i+1}')

    # Plot training data if required
    if T_train is not None:
        plt.plot(T_train, Y_train, 'rx')

    plt.legend()
    plt.title(title)
    plt.show()


def power_normalise(audio_samples):
    # Calculate the Power Spectral Density (PSD)
    psd = np.abs(fft(audio_samples))**2 / len(audio_samples)

    # Compute the scaling factor (reciprocal of the square root of average power)
    scaling_factor = 1 / np.sqrt(np.mean(psd))

    # Apply scaling factor
    normalised_samples = audio_samples * scaling_factor

    return normalised_samples

# ----------------------------------------------------------
# SM kernel spectrum plotting
# ----------------------------------------------------------


def gaussian_function(x, mu, std_dev):
    return 1.0 / (np.sqrt(2.0 * np.pi) * std_dev) * np.exp(-np.power((x - mu) / std_dev, 2.0) / 2)


def return_gaussian(f_spectrum, mu, sig, max_freq=5000, show=False, no_samples=10000):
    # Note the value of the number of samples affects the seen amplitudes (resolution may be too low!)
    output = np.zeros(len(f_spectrum))
    for i in range(len(output)):
        output[i] = gaussian_function(f_spectrum[i], mu=mu, std_dev=sig)
    if show is False:
        return output
    plt.plot(f_spectrum, output)
    plt.show()
    return output


def return_kernel_spectrum(f=[440], M=12, sigma_f=1/500000, show=False, max_freq=10000, no_samples=10000, sample_rate=44100, amplitude=1, B=None, T=None, v=None):
    f_spectrum = np.linspace(0, max_freq, no_samples)
    f_spectrum = fftfreq(no_samples, d=1.0/sample_rate)
    output = np.zeros(len(f_spectrum))
    if v is None:
        v = 2.37
    if T is None:
        T = 0.465
    vertical_lines = []
    for fundamental_frequency in tqdm(f):
        plt.axvline(x=fundamental_frequency, color='green', linestyle='--')
        if B is None:
            closest_key = min(inharmonicity.B.keys(), key=lambda key: abs(
                key - fundamental_frequency))
            B = inharmonicity.B[closest_key]
        for m in tqdm(range(M)):
            inharmonicity_const = np.sqrt((1 + B * (m+1)**2))
            output += 1/(1 + (T*(m+1))**v) * return_gaussian(f_spectrum, fundamental_frequency *
                                                             (m+1) * inharmonicity_const, sigma_f, max_freq=max_freq, no_samples=no_samples)
            vertical_lines.append(fundamental_frequency*(m+1))
    # Times by amplitude scalar
    output = amplitude * output
    plt.plot(f_spectrum, output, label='kernel')
    if len(f) == 1:
        for i, point in enumerate(vertical_lines):
            plt.axvline(x=point, color='pink', linestyle='--')
            plt.text(point, 0.85*max(plt.ylim()),
                     f'M={i+1}', va='bottom', rotation='vertical')
    # Add an information box in the top-left corner
    plt.text(0.8, 1.15, f'T = {T} \nv={v} \nB={B}', transform=plt.gca().transAxes,
             fontsize=12, color='black', verticalalignment='top')
    plt.legend()
    plt.xlim(0, max(f_spectrum)/5)
    if show is False:
        return output, f_spectrum
    plt.show()
    return output, f_spectrum


# ----------------------------------------------------------
# Helper kernel functions and matrices
# ----------------------------------------------------------

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


def RBF_kernel(X1, X2, l=1.0, sigma_f=1/500000,):
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


# ----------------------------------------------------------
# Predictor functions
# ----------------------------------------------------------


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

# ----------------------------------------------------------
# Helper optimiser functions
# ----------------------------------------------------------


def golden_section_f(x1, x2, X_train, Y_train, M=8, sigma=1e-2, tol=1, integer_search=False, B=None):
    # Initial points
    f1 = nlml(X_train, Y_train, M, sigma, f=[x1], B=B)
    f2 = nlml(X_train, Y_train, M, sigma, f=[x2], B=B)

    # Set up golden ratios
    r = (np.sqrt(5)-1)/2.0

    # Third point
    if integer_search is False:
        x3 = x1 * (1-r) + x2 * r
    else:
        x3 = int(x1 * (1-r) + x2 * r)
    f3 = nlml(X_train, Y_train, M, sigma, f=[x3])

    # Loop until convergence
    while abs(x1-x2) > tol:

        if integer_search is False:
            x4 = x1 * r + x2 * (1-r)
        else:
            x4 = int(x1 * r + x2 * (1-r))
        f4 = nlml(
            X_train, Y_train, M, sigma, f=[x4])
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


def golden_section_B(B1, B2, X_train, Y_train, f=[440], M=12, sigma_f=1/500000, tol=0.00008, amplitude=0.000205):
    # Initial points
    f1 = stable_nlml(X_train, Y_train, M=M, sigma_f=sigma_f,
                     f=f, B=B1, w=amplitude)
    f2 = stable_nlml(X_train, Y_train, M=M, sigma_f=sigma_f,
                     f=f, B=B2, w=amplitude)

    # Set up golden ratios
    r = (np.sqrt(5)-1)/2.0

    # Third point
    B3 = B1 * (1-r) + B2 * r
    f3 = stable_nlml(X_train, Y_train, M=M, sigma_f=sigma_f,
                     f=f, B=B3, w=amplitude)

    # Loop until convergence
    while abs(B1-B2) > tol:

        B4 = B1 * r + B2 * (1-r)
        f4 = stable_nlml(
            X_train, Y_train, M=M, sigma_f=sigma_f, f=f, B=B4, w=amplitude)
        print(f4, '/n', B3)
        if f4 < f3:
            B2 = B3
            f2 = f3
            B3 = B4
            f3 = f4
        else:
            B1 = B2
            f1 = f2
            B2 = B4
            f2 = f4
    return B3


def nlml_fn(X_train, Y_train, f=[440], sigma_f=1/500000,  M=12, naive=False):
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
        K = SM_kernel(X_train, X_train, M=M, f=f, sigma_f=theta[0],  w=theta[1]) + \
            0.0005**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
            0.5 * Y_train.dot(inv(K).dot(Y_train)) + \
            0.5 * len(X_train) * np.log(2*np.pi)

    def nlml_stable(theta):
        # Numerically more stable implementation of Eq. (11) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.

        K = SM_kernel(X_train, X_train, M=M, f=f, sigma_f=theta[0],  w=theta[1]) + \
            0.0005**2 * np.eye(len(X_train))
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
