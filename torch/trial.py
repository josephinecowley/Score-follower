# Import the necessary libraries
import math
from matplotlib import pyplot as plt
import torch
import gpytorch
import models
import train
import convenience_functions

# The training data is 15 equally-spaced points from [0,1]
x_train = torch.linspace(0, 10, 200, dtype=torch.float)
frequency = 440
# The true function is sin(2*pi*x) with Gaussian noise N(0, 0.04)
y_train = torch.sin(x_train * (2 * frequency * math.pi)) + \
    torch.randn(x_train.size()) * math.sqrt(0.004)

# Plot training data as black stars
plt.plot(x_train.numpy(), y_train.numpy(), 'k*')
plt.show()

# Initialise the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
model = models.SpectralMixtureGP(x_train, y_train, likelihood)

# Update hyperparameters
hypers = {
    'likelihood.noise_covar.noise': torch.tensor(0.000001),
}
model.initialize(**hypers)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

# Print loglikelihood of prior
loss = -mll(model(x_train), y_train)
print("MLL after training", loss.detach().cpu().item())

# Train model
model, loss = train.train_with_restarts(model, 50, 10)


x_test = torch.linspace(0, 15, 500)
# # plot posterior

observed = convenience_functions.predict(model, likelihood, x_test)
convenience_functions.plot(x_train, y_train, observed, x_test)

convenience_functions.plot_spectral_density(model.spectral_density(model.cov))

loss = -mll(model(x_train), y_train)
print("MLL after training", loss.detach().cpu().item())