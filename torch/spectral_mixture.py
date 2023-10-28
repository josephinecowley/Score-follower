import numpy as np
import math
import torch
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from models import SpectralMixtureGP
import gpytorch
from torch.distributions.mixture_same_family import MixtureSameFamily

# Set default torch dtype to float to allow for MLL approxmation
torch.set_default_dtype(torch.float64)


def plot_spectral_density(density: MixtureSameFamily) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    freq = torch.linspace(
        -density.component_distribution.mean.max()*3, density.component_distribution.mean.max()*3, 5000).reshape(-1, 1)
    x = freq.numpy().flatten()
    y = density.log_prob(freq).numpy().flatten()
    ax.plot(x, y, color="tab:blue", lw=3)
    ax.fill_between(x, y, np.ones_like(x) * y.min(),
                    color="tab:blue", alpha=0.5)
    ax.set_title("Kernel spectral density")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Log Density")
    plt.show()


x_train = torch.linspace(0, 50, 1, dtype=torch.float)
y_train = torch.sin(2 * math.pi * x_train) + torch.sin(6 * math.pi * x_train)

# likelihood = gpytorch.likelihoods.GaussianLikelihood(
#     noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
# model = SpectralMixtureGP(x_train, y_train, likelihood)

# print(model.cov)
# x_train = torch.linspace(0, y_train.size(
#     dim=0) * sample_rate, y_train.size(dim=0))

# x2_train = torch.arange(0, 50, 1)
# # Initialise the likelihood and model


# model.initialize_from_data_empspect(x_train, y_train)
# for param_name, param in model.named_parameters():
#     print(f'Parameter name: {param_name:42} value = {param}')

# hypers = {
#     'likelihood.noise_covar.noise': torch.tensor(100),
#     'covar_module.outputscale': torch.tensor(2.),
#     # 'covar_module.outputscale': torch.tensor(2.),
# }
# plot_spectral_density(model.spectral_density(model.cov))
# model.initialize(**hypers)
# print(
#     model.likelihood.noise_covar.noise,
#     model.covar_module.outputscale.item()
# )
# for param_name, param in model.named_parameters():
#     print(f'Parameter name: {param_name:42} value = {param}')
# plot_spectral_density(model.spectral_density(model.cov))
