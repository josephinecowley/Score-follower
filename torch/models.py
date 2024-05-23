# Import the necessary libraries
import math
from matplotlib import pyplot as plt
import torch
import gpytorch

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal


class SpectralMixtureGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(SpectralMixtureGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean()  # Construct the mean function
        self.cov = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=10, mixture_means_prior=torch.tensor([1/440]))  # Constuct the kernel function
        # Initialise the hyperparameters from the data
        self.cov.initialize_from_data(x_train, y_train)

    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

    def spectral_density(self, smk) -> MixtureSameFamily:
        """Returns the Mixture of Gaussians thet model the spectral density
        of the provided spectral mixture kernel."""
        mus = smk.mixture_means.detach().reshape(-1, 1)
        sigmas = smk.mixture_scales.detach().reshape(-1, 1)
        mix = Categorical(smk.mixture_weights.detach())
        comp = Independent(Normal(mus, sigmas), 1)
        return MixtureSameFamily(mix, comp)
