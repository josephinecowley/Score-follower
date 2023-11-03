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

import inharmonicity


# ----------------------------------------------------------
# Helper plotting functions
# ----------------------------------------------------------

def plot_audio(T, data, show=False):
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
    plt.title("Plot of audio wave")
    if show is True:
        plt.show()


def plot_fft(data, sample_rate=44100, power_spectrum=False, colour='r'):
    """
    Visualise audio frequency data using DFT.

    Args: TODO
        power_spectrum: Make this tru eif plotting a kernel function as the fft will be the power spectrum.
    Returns:
        Returns positive fft data and plots a grah of positive values of the audio spectrum. 
    """
    if power_spectrum is False:
        fft_data = abs(fft(data, norm="ortho"))

    else:
        fft_data = np.sqrt(abs(fft(data, norm="ortho")))
    frequency_axis = fftfreq(len(data), d=1.0/sample_rate)
    plt.plot(frequency_axis[:(len(data)//8)],
             fft_data[:(len(data)//8)], colour, label='Audio')
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplitude')
    plt.title('Spectrum')


def plot_kernel(T, kernel):
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
    plt.title("Plot of kernel function")
    plt.show()


def plot_kernel_matrix(kernel_matrix):
    """
    Visualise kernel matrix.

    Args:
        kernel_matrix: a matrix.

    Returns:
        Plot of matrix in heat map. 
    """
    plt.imshow(kernel_matrix, cmap='coolwarm', interpolation='nearest')
    plt.title("Covariance Matrix Heatmap")
    plt.colorbar()
    plt.show()


def plot_gp(mu, cov, T_test, T_train=None, Y_train=None, samples=0):
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

    # Generate and plot random samples
    for i in range(samples):
        # Draw three samples from the prior
        z = np.random.randn(len(T_test), 1).ravel()
        # Add a small amount of noise to ensure it is ositive definite
        K = cov + 1e-6 * np.eye(len(T_test))
        L = np.linalg.cholesky(K)
        y = np.dot(L.T, z) + mu
        plt.plot(T_test, y, lw=1, ls='--', label=f'Sample {i+1}')

    # Plot training data if required
    if T_train is not None:
        plt.plot(T_train, Y_train, 'rx')

    plt.legend()
    plt.show()

# ----------------------------------------------------------
# SM kernel spectrum plotting
# ----------------------------------------------------------


def gaussian_function(x, mu, sig):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)


def return_gaussian(mu, sig, max_freq=5000, show=False, no_samples=10000):
    # Note the value of the number of samples affects the seen amplitudes (resolution may be too low!)
    f_spectrum = np.linspace(0, max_freq, no_samples)
    output = np.zeros(len(f_spectrum))
    for i in range(len(output)):
        output[i] = gaussian_function(f_spectrum[i], mu=mu, sig=sig)
    if show is False:
        return output
    plt.plot(f_spectrum, output)
    plt.show()
    return output


def return_kernel_spectrum(f=[440], M=6, sigma_f=10, show=False, max_freq=5000, no_samples=10000, scalar=1):
    f_spectrum = np.linspace(0, max_freq, no_samples)
    output = np.zeros(len(f_spectrum))
    v = 1.4
    T = 1.5
    vertical_lines = []
    for fundamental_frequency in f:
        for m in range(M):
            B = inharmonicity.B[int(fundamental_frequency)]
            inharmonicity_const = np.sqrt((1 + B * (m+1)**2))
            output += scalar*1/(1 + (T*(m+1))**v) * return_gaussian(fundamental_frequency *
                                                                    (m+1) * inharmonicity_const, sigma_f, max_freq=max_freq, no_samples=no_samples)
            vertical_lines.append(fundamental_frequency*(m+1))
    for i, point in enumerate(vertical_lines):
        plt.axvline(x=point, color='pink', linestyle='--')
        plt.text(point, 0.85*max(plt.ylim()),
                 f'M={i}', va='bottom', rotation='vertical')
    if show is False:
        plt.plot(f_spectrum, output, label='kernel')
        # Add a legend
        # Add an information box in the top-left corner
        plt.text(0.8, 1.1, f'T = {T} \nv={v}', transform=plt.gca().transAxes,
                 fontsize=12, color='black', verticalalignment='top')
        plt.legend()

        return output, f_spectrum
    plt.plot(f_spectrum, output)
    plt.show()
    return output, f_spectrum


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
    v = 0.5
    cosine_series = 0
    for fundamental_frequency in f:
        for m in range(M):
            B = inharmonicity.B[int(fundamental_frequency)]
            inharmonicity_const = np.sqrt((1 + B * (m+1)**2))
            cosine_series += 1/(1 + (T*(m+1))**v) * np.cos((m+1) * 2 * np.pi *
                                                           fundamental_frequency * inharmonicity_const * np.linalg.norm(t1 - t2))
    return np.exp(-(sigma_f**2) * 2 * np.pi**2 * np.linalg.norm(t1 - t2)**2) * cosine_series


def return_SM_kernel(T, show=False, M=6, f=[440], sigma_f=1e-5):
    """
    Return kernel function.
    Automatically does not show function, unless show=True.

    Args:
        T: 1D numpy array of time samples.
        show: If True, also display kernel function. If False, only return kernel.

    Returns:
        Kernel values of T as numpy array. 
    """
    kernel = np.zeros(len(T))
    for i in range(len(T)):
        kernel[i] = SM_kernel(0, T[i], M=M, f=f, sigma_f=sigma_f)
    if show is False:
        return kernel
    plot_kernel(T, kernel)
    return kernel


def return_SM_kernel_matrix(T1, T2=None, M=6, f=[440], sigma_f=1e-5, show=False):
    """
    Compute the Spectral Mixture kernel matrix between two sets of vectors.
    If T2 is None, find matrix between T1 and T1.
    Automatically does not show matrix, unless show=True.

    Args:
        T1: Array of 1D time samples of length n1.
        T2: Array of 1D time samples of length n2.
        show: if True, plot heat map of covariance function.

    Returns:
        Matrix (n1, n2) of scalars from SM kernel.TODO check this retirns thjezse dimensions
    """
    if T2 is None:
        T2 = T1

    n1 = len(T1)
    n2 = len(T2)

    kernel_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            t1 = T1[i]
            t2 = T2[j]

            kernel_matrix[i, j] = SM_kernel(t1, t2, M=M,  f=f, sigma_f=sigma_f)

    if show is False:
        return kernel_matrix
    plot_kernel_matrix(kernel_matrix)
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


def nlml(T, Y, cov_s=None, M=10, sigma_f=100, f=[400], sigma_n=1e-2):
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
    if cov_s is None:
        K = return_SM_kernel_matrix(T, T, M=M, sigma_f=sigma_f, f=f) + \
            sigma_n**2 * np.eye(len(T))
    else:
        K = cov_s + sigma_n**2 * np.eye(len(T))
    return 0.5 * Y.dot(inv(K).dot(Y)) + 0.5 * np.log(det(K)) + 0.5 * len(T) * np.log(2*np.pi)


def stable_nlml(T, Y,  M=8, sigma_f=20, f=[440], sigma_n=1e-2):
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
    K = return_SM_kernel_matrix(T, T, M=M, sigma_f=sigma_f, f=f) + \
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


def posterior(T_test, T_train, Y_train, M=8, sigma_f=20, sigma_y=0.005, f=[440]):
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
    K = return_SM_kernel_matrix(T_train, T_train, M=M, sigma_f=sigma_f, f=f) + \
        sigma_y**2 * np.eye(len(T_train))
    K_s = return_SM_kernel_matrix(
        T_train, T_test, M=M, sigma_f=sigma_f, f=f)
    K_ss = return_SM_kernel_matrix(
        T_test, T_test, M=M, sigma_f=sigma_f, f=f) + 1e-8 * np.eye(len(T_test))
    K_inv = inv(K)

    # Calculate posterior mean function
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Calculate posterior covariance function
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s

# ----------------------------------------------------------
# Helper optimiser functions
# ----------------------------------------------------------


def golden_section(x1, x2, X_train, Y_train, M=8, sigma=1e-2, tol=1, integer_search=False):
    # Initial points
    f1 = nlml(X_train, Y_train, M, sigma, f=[x1])
    f2 = nlml(X_train, Y_train, M, sigma, f=[x2])

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


def nlml_fn(X_train, Y_train, M=8, naive=False):
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
        K = return_SM_kernel_matrix(X_train, X_train, M=M, f=[theta[0]], sigma_f=theta[1]) + \
            theta[2]**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
            0.5 * Y_train.dot(inv(K).dot(Y_train)) + \
            0.5 * len(X_train) * np.log(2*np.pi)

    def nlml_stable(theta):
        # Numerically more stable implementation of Eq. (11) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.

        K = return_SM_kernel_matrix(X_train, X_train, M=M, f=[theta[0]], sigma_f=theta[1]) + \
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
