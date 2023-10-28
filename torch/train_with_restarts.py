# Import the necessary libraries
import math
from matplotlib import pyplot as plt
import torch
import gpytorch
from models import SpectralMixtureGP
import convenience_functions

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

# Generating the data

# The training data is 15 equally spaced points from [0,1]
x_train = torch.linspace(0, 1, 15)

# The true function is sin(2*pi*x) with Gaussian noise N(0, 0.04)
y_train = torch.sin(x_train*2*math.pi) + \
    torch.randn(x_train.size()) * math.sqrt(0.04)

# Plot the training data
plt.plot(x_train.numpy(), y_train.numpy(), "*k")
# plt.show()


# Initialise the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SpectralMixtureGP(x_train, y_train, likelihood)

# Training the model


# Put the model into training mode
model.train()
likelihood.train()

convenience_functions.train(x_train, y_train, model,
                            likelihood, training_iter=50)


# Making predictions with the model

# The test data is 50 equally-spaced points from [0,5]
x_test = torch.linspace(0, 5, 50)

# Put the model into evaluation mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Obtain the predictibe mean and covariance matrix
    f_preds = model(x_test)
    f_mean = f_preds.mean
    f_cov = f_preds.covariance_matrix

    # Make predictions by feeding model through likelihood
    observed_pred = likelihood(model(x_test))

    # Initialise plot
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x_train.numpy(), y_train.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(x_test.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()
