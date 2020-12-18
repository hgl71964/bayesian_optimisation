"""
2020 Summer internship

To evaluate a GP
"""

import torch
import torch.nn as nn
import os

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP
from botorch.models import SingleTaskGP
import gpytorch
from sklearn.model_selection import train_test_split

class gp_evaluation:

    @staticmethod
    def evaluation(x, y):
        """
        for a specific kernel, we train on 3 training set, with 20, 50, 80 training samples respectively, 
                                        and then average over their results over 3 runs
        """

        kernels = ["SE", "RQ", "MA2.5", "PO2", "PO3", "LR"]
        modes = ["raw", "add", "pro"]
        opts = ["ADAM", "quasi_newton"]
        records = [None] * (len(kernels) * len(modes) * len(opts))

        mse = nn.MSELoss()
        count = 0

        for kernel in kernels:
            for mode in modes:
                for opt in opts:
                    temp_loss = 0
                    temp_mse = 0
                    for train_sample in [20, 50, 80]:
                        X_train, y_train, X_test, y_test = gp_evaluation.make_data(x, y, train_sample)
                        k = gp_evaluation.make_kernel(kernel)
                        mll, m = gp_evaluation.instance_gp(X_train, y_train, k, mode)

                        if opt == "ADAM":
                            gp_evaluation.ADAM(mll, m, X_train, y_train)
                        elif opt == "quasi_newton":
                            gp_evaluation.quasi_newton(mll)
                        
                        m.eval()
                        m.likelihood.eval()
                        output = m(X_test)

                        try:
                            loss = mll(output, y_test)
                            out = m(X_test).mean.detach()
                        except:
                            temp_loss += -float("inf")  # error because of chlosky decomposition, reflecting this kernel is not stable
                        else:
                            temp_loss += loss.item()  # normal
                        temp_mse += float(mse(out, y_test))
                    
                    records[count] = (kernel, mode, opt, temp_mse/3, temp_loss/3)
                    count+=1
        return records
    
    @staticmethod
    def result_and_save(records, data_folder, portfolio_number: int):
        full_path = os.path.join(data_folder, f"portfolio {portfolio_number}, gp_evaluation_results")
        with open(full_path + ".pt", "wb") as f:
            torch.save(records, f)
            f.close()
        
        return sorted(records, key = lambda x:x[-1])[-1]  #  best kernel; (kernel, mode, opt, temp_mse, temp_loss)

    @staticmethod
    def make_data(X, y, train_sample = 50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=0.75, random_state=42)
        X_train, X_test, y_train, y_test = X_train.float(), X_test.float(), \
                                            y_train.float(), y_test.float()
        #  down-sampling
        X_train = X_train[:train_sample]
        y_train = y_train[:train_sample]        # train; y_train -> shape[n, 1]
        y_test = y_test.squeeze(-1)             # test; y_test -> shape[n,]

        return X_train, y_train, X_test, y_test


    @staticmethod
    def ADAM(mll, model, x, y):
        """
        MLE tuning via ADAM
        Args:
            x -> shape[n,d]; tensor
            y -> shape[n,1]; tensor
        """
        y = y.squeeze(-1)
        model.train()
        model.likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        epochs = 128

        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
        return None
    
    @staticmethod
    def quasi_newton(mll):
        fit_gpytorch_model(mll)

    @staticmethod
    def make_kernel(name):
        if name == "SE":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif name == "RQ":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        elif name == "MA2.5":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        elif name == "PO2":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power = 2))
        elif name == "PO3":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power = 3))
        elif name == "LR":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        return kernel
    
    @staticmethod
    def instance_gp(x, y, kernel, mode):
        # zeros-noise settings
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = 1e-4  
        likelihood.noise_covar.raw_noise.requires_grad_(False)
        m = SingleTaskGP(x, y, likelihood)

        if mode == "raw":
            m.covar_module = kernel
        elif mode == "add":
            m.covar_module = gpytorch.kernels.AdditiveStructureKernel(base_kernel = kernel, num_dims = x.size(-1))
        elif mode == "pro":
            m.covar_module = gpytorch.kernels.ProductStructureKernel(base_kernel = kernel, num_dims = x.size(-1))

        m = m.float()
        m.likelihood = m.likelihood.float()
        mll = ExactMarginalLogLikelihood(m.likelihood, m)
        return mll, m


    @staticmethod
    def simple_evaluation(x, y):
        """
        for a specific kernel, we train on 3 training set, with 20, 50, 80 training samples respectively, 
                                        and then average over their results over 3 runs
        """

        kernels = ["MA2.5"]
        modes = ["raw"]
        opts = ["ADAM", "quasi_newton"]
        records = [None] * (len(kernels) * len(modes) * len(opts))

        mse = nn.MSELoss()
        count = 0

        for kernel in kernels:
            for mode in modes:
                for opt in opts:
                    temp_loss = 0
                    temp_mse = 0
                    for train_sample in [20, 50, 80]:
                        X_train, y_train, X_test, y_test = gp_evaluation.make_data(x, y, train_sample)
                        k = gp_evaluation.make_kernel(kernel)
                        mll, m = gp_evaluation.instance_gp(X_train, y_train, k, mode)

                        if opt == "ADAM":
                            gp_evaluation.ADAM(mll, m, X_train, y_train)
                        elif opt == "quasi_newton":
                            gp_evaluation.quasi_newton(mll)
                        
                        m.eval()
                        m.likelihood.eval()
                        output = m(X_test)

                        try:
                            loss = mll(output, y_test)
                            out = m(X_test).mean.detach()
                        except:
                            temp_loss += -float("inf")  # error because of chlosky decomposition, reflecting this kernel is not stable
                        else:
                            temp_loss += loss.item()  # normal
                        temp_mse += float(mse(out, y_test))
                    
                    records[count] = (kernel, mode, opt, temp_mse/3, temp_loss/3)
                    count+=1
        return records