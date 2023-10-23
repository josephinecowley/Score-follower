import numpy as np
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


# Wav file method
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/tuner_440.wav'

# Read a Wav file
sample_rate, y_train = wav.read(wav_file)
y_train = torch.from_numpy(y_train[:300])
x_train = torch.linspace(0, y_train.size(
    dim=0) * sample_rate, y_train.size(dim=0))

x2_train = torch.arange(0, 50, 1)
# Initialise the likelihood and model


likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-6))


model = SpectralMixtureGP(x2_train, x_train, likelihood)
for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param}')

hypers = {
    'likelihood.noise_covar.noise': torch.tensor(100),
    'covar_module.outputscale': torch.tensor(2.),
    # 'covar_module.outputscale': torch.tensor(2.),
}
plot_spectral_density(model.spectral_density(model.cov))
model.initialize(**hypers)
print(
    model.likelihood.noise_covar.noise,
    model.covar_module.outputscale.item()
)
for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param}')
plot_spectral_density(model.spectral_density(model.cov))
