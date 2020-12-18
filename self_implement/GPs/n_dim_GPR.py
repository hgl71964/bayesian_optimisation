"""
2020 Summer internship

This scripts provides a Gaussian process regression on n-dimensional space, 
    used as probabilistic surrogate model
"""

import numpy as np

class _ndim_GPR:


    def __init__(self,**kwargs):
        """
        kwargs:
            hyperparameter for the gaussian process

            such as:
                {'kernel_type': 'SE',
                'len_scale': 1e-2,
                'amplify_coef':1e-1,
                }
        """
        self.kernel_type = kwargs.get("kernel_type")
        self.len_scale = kwargs.get("len_scale")
        self.amplify_coef = kwargs.get("amplify_coef")

        # assign hyper-parameter to class attributes
        # for key,value in kwargs.items():
        #     setattr(self,f"{key}",value)


    def _cov_matrix(self,x,t):
        """
        Goal:
            compute Gaussian covariance (kernel_type) matrix

        Args:
            x: np.2darray; shape: [k,n]
            t: np.2darray; shape: [l,n]

        Returns:
            covariance matrix: np matrix; [k,l]
        """

        if self.kernel_type == "SE":
            """
            SE covariance matrix:
                k(x,t) = amplify_coef^2 * exp[ -(x-t)^2/ 2l^2 ]

            hyperparameter:
                amplify_coef -> amplification coefficient
                l -> characteristic length coefficient
            """

            row = x.shape[0]
            col = t.shape[0]

            factor_matrix = np.einsum("ij,kj->ik",x,t) ** 2 # key
            
            assert factor_matrix.shape == (row,col) # check 
            
            cov_mat = self.amplify_coef * np.exp(factor_matrix/(-2*(self.len_scale**2)))

        return cov_mat


    def gp_posterior(self,x,y,X_test,mean_function,mean_function_test):
        """
        Goal:
            obtain conditional gaussian distribution, i.e. posterior distribution 

        Args:
            x: existing indices ->  np.2d array: [m,n]
            y: existing observation (function values) -> np.2d array: [m,]

            X_test: to 'smooth out' the posterior mean -> np.1d array: [N,n]
            mean_function: mean of x
            mean_function: mean of X_test

        Returns:
            posterior mean -> np.1darray [N,]
            posterior covariance (squared) -> np matrix: [N,N]
        """
        Σ11 = self._cov_matrix(x, x)
        Σ12 = self._cov_matrix(x, X_test)
        Σ21 = Σ12.T
        Σ22 = self._cov_matrix(X_test, X_test)

        Σ11_inverse = np.linalg.pinv(Σ11)

        posterior_mean = mean_function_test + Σ21 @ Σ11_inverse @ (y.reshape(-1,1) - mean_function)

        posterior_cov = Σ22 - Σ21 @ Σ11_inverse @ Σ12

        return posterior_mean.flatten(), posterior_cov