# Import the necessary libraries
import math
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
import torch
import gpytorch
import models
import train
import convenience_functions

# The training data is 15 equally-spaced points from [0,1]
# x_train = torch.linspace(0, 10, 200, dtype=torch.float)
# frequency = 440
# # The true function is sin(2*pi*x) with Gaussian noise N(0, 0.04)
# y_train = torch.sin(x_train * (2 * frequency * math.pi)) + \
#     torch.randn(x_train.size()) * math.sqrt(0.004)

# # Read a Wav file
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/tuner_440.wav'
sample_rate, y_train = wav.read(wav_file)
y_train = torch.Tensor(y_train[:200])
x_train = torch.linspace(0, y_train.size(
    dim=0) * sample_rate, y_train.size(dim=0))

# Plot training data as black stars
plt.plot(x_train.numpy(), y_train.numpy(), 'k*')
plt.show()


# Initialise the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-5))
model = models.SpectralMixtureGP(x_train, y_train, likelihood)
# model.cov.mixture_means = torch.tensor([[325.8866], [4.4211], [440.0000]])
# model.cov.mixture_scales = torch.tensor([[5.0000], [0.0005], [0.000005]])
# model.cov.mixture_weights = torch.tensor([[0.1426], [0.1246], [0.126]])
convenience_functions.plot_spectral_density(model.spectral_density(model.cov))
print(model.cov.mixture_means.detach().reshape(-1, 1),
      model.cov.mixture_scales.detach().reshape(-1, 1), model.cov.mixture_weights.detach())
# Update hyperparameters
hypers = {
    'likelihood.noise_covar.noise': torch.tensor(0.01)
}
model.initialize(**hypers)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

# Print loglikelihood of prior
loss = -mll(model(x_train), y_train)
# print("MLL after training", loss.detach().cpu().item())

# Train model
model, loss = train.train_with_restarts(model, 50, 10)


x_test = torch.linspace(0, 15, 500)
# # plot posterior

observed = convenience_functions.predict(model, likelihood, x_test)
convenience_functions.plot(x_train, y_train, observed, x_test)

convenience_functions.plot_spectral_density(model.spectral_density(model.cov))

loss = -mll(model(x_train), y_train)
print("MLL after training", loss.detach().cpu().item())
