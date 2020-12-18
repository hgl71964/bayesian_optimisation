"""
2020 Summer internship

This sript implements 1d Gaussian Process for the ease of visualisation

"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import copy


class _GP_regression:

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
            x, t: np 1d array

        Returns:
            covariance matrix: np matrix; [len(x),len(t)]
        """

        if self.kernel_type == "SE":
            """
            SE covariance matrix:
                k(x,t) = amplify_coef^2 * exp[ -(x-t)^2/ 2l^2 ]

            hyperparameter:
                amplify_coef -> amplification coefficient
                l -> characteristic length coefficient
            """

            row = len(x)
            col = len(t)
            x = np.repeat(x.reshape(-1,1),col,axis=1)
            t = np.repeat(t.reshape(1,-1),row,axis=0)
            cov_mat = self.amplify_coef * np.exp(((x-t)**2)/(-2*(self.len_scale**2)))

        return cov_mat


    def gp_posterior(self,x,y,X_test,mean_function,mean_function_test):
        """
        Goal:
            obtain conditional gaussian distribution, i.e. posterior distribution 

        Args:
            x: existing indices ->  np.1d array: [m,]
            y: existing observation (function values) -> np.1d array: [m,]

            X_test: to 'smooth out' the posterior mean -> np.1d array: [n,]
            mean_function: mean of x
            mean_function: mean of X_test

        Returns:
            posterior mean -> np.1darray [n,]
            posterior covariance (squared) -> np matrix: [n,n]
        """
        Σ11 = self._cov_matrix(x, x)
        Σ12 = self._cov_matrix(x, X_test)
        Σ21 = Σ12.T
        Σ22 = self._cov_matrix(X_test, X_test)

        Σ11_inverse = np.linalg.pinv(Σ11)

        posterior_mean = mean_function_test + Σ21 @ Σ11_inverse @ (y.reshape(-1,1) - mean_function)

        posterior_cov = Σ22 - Σ21 @ Σ11_inverse @ Σ12

        return posterior_mean.flatten(), posterior_cov
    
    def visualize(self,x,y,X_test,mean_function,mean_function_test,N_func_sample = 5):
        """
        draw function samples from posterior mean and std

        Args:
            as per gp_posterior
            N_func_sample: number of function samples drawn from the posterior 
        """
        pst_mean, pst_Sigma = self.gp_posterior(x,y,X_test,mean_function,mean_function_test)
        pst_std = np.sqrt(np.diag(pst_Sigma)).flatten()

        # function samples
        sample_y = np.random.multivariate_normal(mean=pst_mean, cov=pst_Sigma, size=N_func_sample)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

        ax.set_ylabel('y',fontsize = 10)
        ax.set_xlabel('x',fontsize = 10)
        ax.set_title('Gaussian process posterior',fontsize = 14)
        # ax.set_ylim(-2,2)
        ax.set_xlim(0,1)

        ax.scatter(x,y,marker='o',
            linewidth=4,color='black', label='samples', alpha=0.9)

        ax.plot(X_test,pst_mean,linewidth=2,color='blue', label='posterior mean', alpha=1)

        ax.plot(X_test, sample_y.T, '-', alpha=0.3)

        ax.fill_between(X_test, pst_mean-2*pst_std, pst_mean+2*pst_std, color='red', 
                        alpha=0.2, label='$2 \sigma_{2|1}$')
        plt.show()