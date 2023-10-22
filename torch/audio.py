import math
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
import torch
import gpytorch

# Set default torch dtype to float to allow for MLL approxmation
torch.set_default_dtype(torch.float64)

# Wav file method
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/tuner_440.wav'

# Read a Wav file
sample_rate, y_train = wav.read(wav_file)
y_train = torch.from_numpy(y_train[:300])
x_train = torch.linspace(0, y_train.size(
    dim=0) * sample_rate, y_train.size(dim=0))

# Plot the training data
plt.plot(x_train.numpy(), y_train.numpy(), "*k")
plt.show()


# A GP model with constant mean function and SM kernel

class SpectralMixtureGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(SpectralMixtureGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean()  # Construct the mean function
        self.cov = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=10, mixture_means=[440, 880, 1320, ])  # Constuct the kernel function
        # Initialise the hyperparameters from the data
        self.cov.initialize_from_data(x_train, y_train)

    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)


# Initialise the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SpectralMixtureGP(x_train, y_train, likelihood)

# Training the model


# Put the model into training mode
model.train()
likelihood.train()


# Use the Adam optimiser, with learning rate set to 0.1
optimiser = torch.optim.Adam(model.parameters(), lr=0.1)

# Use the negative marhinal log-likelihood as the loss function
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Set the number of training iterations
n_iter = 50

for i in range(n_iter):
    # Set the gradients from previous iteration to zero
    optimiser.zero_grad()
    # Ouput from the model
    output = model(x_train)
    # Compute loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
    optimiser.step()


# Making predictions with the model


# The test data is 5 times the length of the training data, at equally-spaced points from [0,5]
x_test = torch.linspace(0, 2*y_train.size(
    dim=0) * sample_rate, 500)

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
