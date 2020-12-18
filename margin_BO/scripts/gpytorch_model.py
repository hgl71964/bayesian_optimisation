"""
2020 Summer internship

implement Gpytorch model, such that it can be used by BOtorch
"""

import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch import fit_gpytorch_model

class GPytorch_GP:
    """
    data type assume torch.tensor.double()
    """
    def __init__(self,gp_name,gp_params):
        self.name = gp_name
        self.gp_params = gp_params
    
    def init_model(self,x,y,state_dict=None):
        """
        initialise and update a custom Gpytorch model every outside loop

        Args:
            x: training samples; shape [n, d] -> n samples, d-dimensional
            y: function values; shape [n,1] -> 1-dimensional-output
            state_dict: update model when it is provided

        Returns:
            mll: Gpytorch Marginal likelihood
            model: Gpytorch model
        """
        # change data format since it's different for Gpytorch model
        y = y.squeeze(-1)

        # make zero noise
        noises = torch.zeros_like(y,dtype=y.dtype)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noises,learn_additional_noise=False)
        
        model = eval(f"{self.name}"+"_GP(x,y,likelihood,**self.gp_params)")

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model

    def fit_model(self, mll):
        """
        MLE tuning via L-BFGS-B
        mll: marginal likelihood of model; work for BOtorch and Gpytorch model
        """

        # TODO: add other methods
        fit_gpytorch_model(mll)

    def _SM(self,x,y):
        kernel = gpytorch.kernels.SpectralMixtureKernel(
                            num_mixtures=self.params.get("num_mixtures"),
                            ard_num_dims=x.size(-1),
                            batch_shape= torch.Size([256]),
                            )

        """
        this kernel must contain at least two data point
            (the documentation says init kernel from empirical data specturm works better)

        initialize_from_data; x->[n,d], y -> [n,1]
        initialize_from_data_empspect; x->[n,d], y -> [n,]
        """
        # print(x.size(),y.size())

        kernel.initialize_from_data(x, y)
        # y=y.squeeze(-1)
        # kernel.initialize_from_data_empspect(x, y) 
        return kernel

    def __str__(self):
        return self.name
