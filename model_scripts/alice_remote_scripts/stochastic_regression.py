'''
import torch
import pyro
from pyro.distributions import Dirichlet
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive

# Define the model
class DirichletRegressionModel(pyro.nn.PyroModule):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = pyro.nn.PyroModule[torch.nn.Linear](input_size, output_size)
        self.linear.weight = pyro.nn.PyroSample(dist.Normal(0., 1.).expand([output_size, input_size]).to_event(2))
        self.linear.bias = pyro.nn.PyroSample(dist.Normal(0., 1.).expand([output_size]).to_event(1))

    def forward(self, X, y=None):
        alpha = torch.exp(self.linear(X))  # Ensure alpha is positive
        with pyro.plate('data', X.shape[0]):
            obs = pyro.sample("obs", Dirichlet(alpha), obs=y)
        return obs


def main_dr():
    # Initialize the model
    input_size = 728  # Size of embedding
    output_size = 3  # Number of sentiment levels
    model = DirichletRegressionModel(input_size, output_size)

    # Use Pyro's AutoDiagonalNormal guide
    guide = AutoDiagonalNormal(model)

    # Prepare data
    # X: embeddings, y: probability vectors
    #X = torch.tensor(embeddings, dtype=torch.float)
    #y = torch.tensor(prob_vectors, dtype=torch.float)

    # Setup the optimizer and inference algorithm
    optim = Adam({"lr": 0.03})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    # Training loop
    num_iterations = 1000
    for i in range(num_iterations):
        loss = svi.step(X, y)
        if i % 100 == 0:
            print(f"Step {i}, loss = {loss}")

    # Save the learned parameters
    pyro.clear_param_store()
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    # Inference on new data X_new
    predictive = Predictive(model, guide=guide, num_samples=1000, return_sites=("obs",))
    samples = predictive(X_new)

    # Calculating mean and standard deviation of predictions
    mean_predictions = samples["obs"].mean(dim=0)
    std_predictions = samples["obs"].std(dim=0)

'''
#------------------------ Gaussian Processes --------------------------

import torch
import torch.nn as nn
import gpytorch

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model

# Assume that `input_data` is a tensor of integer indices representing sentences
embedded_data = embedding_model(input_data)

model = GPRegressionModel(embedded_data, train_y, likelihood)


