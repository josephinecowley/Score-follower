import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from torch.distributions.mixture_same_family import MixtureSameFamily


import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50


def train(x_train, y_train, model, likelihood, training_iter=training_iter):
    # Put the model into training mode
    model.train()
    likelihood.train()
    # Use the adam optimizer
    # Includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()


def predict(model, likelihood, test_x=torch.linspace(0, 1, 51)):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Test points are regularly spaced along [0,1]
        return likelihood(model(test_x))


def plot(x_train, y_train, observed_pred, x_test):
    with torch.no_grad():
        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(x_train.numpy(), y_train.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(x_test.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(x_test.numpy(), lower.numpy(),
                        upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()


def plot_spectral_density(density: MixtureSameFamily) -> None:
    fix, ax = plt.subplots(figsize=(9, 6))
    freq = torch.linspace(
        0, 3*density.component_distribution.mean.max(), 5000).reshape(-1, 1)
    x = freq.numpy().flatten()
    y = density.log_prob(freq).numpy().flatten()
    ax.plot(x, y, color="tab:blue", lw=3)
    ax.fill_between(x, y, np.ones_like(x) * y.min(),
                    color="tab:blue", alpha=0.5)
    ax.set_title("Kernel spectral density")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Log Density")
    plt.show()


def plot_kernel(kernel, ax, xx=torch.linspace(-0.1, 0.1, 1000), col="tab:blue"):
    x0 = torch.zeros(xx.size(0))
    ax.plot(xx.numpy(), np.diag(kernel(xx, x0).numpy()), color=col)
