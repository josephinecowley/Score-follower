import math
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
import torch
import gpytorch
import convenience_functions
from models import SpectralMixtureGP
import train

# Set default torch dtype to float to allow for MLL approxmation
torch.set_default_dtype(torch.float64)


# Read a Wav file
wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/Sine.wav'
sample_rate, y_train = wav.read(wav_file)
y_train = torch.Tensor(y_train[:500])
x_train = torch.linspace(0, y_train.size(
    dim=0) * sample_rate, y_train.size(dim=0))

# Plot the training data
plt.plot(x_train.numpy(), y_train.numpy(), "*k")
plt.show()
print((x_train.dtype), (y_train.dtype))

# Initialise the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
model = SpectralMixtureGP(x_train, y_train, likelihood)


# Update hyperparameters
hypers = {
    'likelihood.noise_covar.noise': torch.tensor(0.00001),
}
model.initialize(**hypers)
# model.cov.mixture_means = torch.tensor([[0.000066], [0.0001211]])
# model.cov.mixture_scales = torch.tensor([[5.0000], [0.0005], [0.000005]])
# model.cov.mixture_weights = torch.tensor([[0.1426], [0.1246], [0.126]])


def plot_model_kernel():
    fig, ax = plt.subplots(figsize=(9, 7))
    convenience_functions.plot_kernel(
        model.cov, xx=torch.linspace(-2, 2, 1000), ax=ax, col="tab:blue")
    ax.set_title("Learned kernel")
    plt.show()


convenience_functions.plot_spectral_density(
    model.spectral_density(model.cov))
model, loss = train.train_with_restarts(
    model=model,
    num_iters=50,
    num_restarts=2,
    lr=0.1,
    show_progress=True,
)
print(loss)

# The test data is 5 times the length of the training data, at equally-spaced points from [0,5]
x_test = torch.linspace(0, 2*y_train.size(
    dim=0) * sample_rate, 1000)


observed_pred = convenience_functions.predict(model, likelihood, test_x=x_test)
convenience_functions.plot(x_train, y_train, observed_pred, x_test)

for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param}')
print(model.cov)
convenience_functions.plot_spectral_density(
    model.spectral_density(model.cov))
plot_model_kernel()
print(model.cov.mixture_means.detach().reshape(-1, 1))
