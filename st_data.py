import math
import torch
import gpytorch

train_x = None
train_y = None

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# We want to initialize the actual parameter, not the raw parameters
# see https://github.com/cornellius-gp/gpytorch/blob/master/examples/00_Basic_Usage/Hyperparameters.ipynb

hypers = {
    # We won't be using the likelihood so we don't need to initialize the noise
    'covar_module.base_kernel.lengthscale': torch.tensor([1.0, 1.0, 1.0]),
    'covar_module.outputscale': torch.tensor(2.0),
}

model.initialize(**hypers)

N = 1000
test_x = torch.randn(N, 3)

model.eval()
with gpytorch.settings.prior_mode(True):
    test_y = model(test_x).sample()
    print(test_y.shape)