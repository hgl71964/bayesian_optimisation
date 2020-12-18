"""
2020 Summer internship

This sript implements Expected Improvement to use as acquisition function

Warnings:
    this acquisition function only accept 1d input and thus is only for visualisation
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import copy

"""
from scipy.stats use CDF and PDF of standard normal distribution
"""
import scipy.stats

"""
import quasi-newton minimiser 
"""
from scipy.optimize import minimize


class _expected_improvement:
    """
    expected improvement used by bayes opter

    have three functions:

    1. compute_func: used to compute the values of function

    2. propose_position: returns the maximum of acquisition function and its x

    3. plot_func: plot acquisition function
    """

    def __init__(self,**kwargs):
        
        self.xi = kwargs.get("xi")
        del kwargs["xi"]
        # self.gpr = _GP_regression(**kwargs)     

    def compute_func(self,model,x,y,X_test,mean_function=0,mean_function_test=0):
        """
        compute expected improvement function

        Args:
            x: observed indices [m,]
            y: observed values [m,]
            X_test: test indices to 'smooth out' posterior [n,]

        Returns:
            ei: expected improvement values along test points
                -> np.1darray [n,]
        """

        pst_μ, pst_Σ=model.gp_posterior(x,y,X_test,mean_function,mean_function_test)
        pst_σ = np.sqrt(np.diag(pst_Σ)).flatten()
        f_star = np.max(y)
        # ei = np.zeros(len(X_test))

        """
        implement expected improvement
        """

        with np.errstate(divide="warn"):
            improvement = pst_μ - f_star - self.xi
            Z = improvement / pst_σ 
            ei = improvement * scipy.stats.norm.cdf(Z) +\
                pst_σ * scipy.stats.norm.pdf(Z)
            ei[pst_σ == 0.] = 0.
        
        """
        handle divide by 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide( np.arange(3), 0 )
            c[ ~ np.isfinite( c )] = 0 
        """

        return ei

    
    def propose_position(self,model,bounds,opt_restart,x,y,X_test,mean_function=0,mean_function_test=0):
        """
        x* = argmax [acquisition function], i.e. maximise acquisition function

        Args:
            x,y, X_test: see EI

            bounds: range of variable x -> np.2darray
            opt_restart: number of random start in L-BFGS-B -> int

        Returnes:
            -min_val: maximum acquisition function value
            min_x: (scalar) 
        """
        min_val = float('inf') 
        min_x = None

        # maximise acquisition function = minimise its negative (scipy has only minimiser)
        obj_func = lambda query: -1 * self.compute_func(model,x,y,query,mean_function,mean_function_test)

        """
        randomly restart opt_restart times to maximise acquisition function

        x0 = initial guess/query
        """
        for x0 in np.random.uniform(bounds[:,0],bounds[:,1],size=(opt_restart,1)):
            res = minimize(obj_func,x0=x0,bounds=bounds,method="L-BFGS-B")
            if res.fun[0] < min_val:
                min_val = res.fun[0]
                min_x = res.x[0]
        return -min_val, min_x

    def plot_func(self,model,x,y,X_test,mean_function=0,mean_function_test=0):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        ax.set_ylabel('y',fontsize = 14)
        ax.set_xlabel('x',fontsize = 14)
        ax.set_title('Acquisition Function',fontsize = 20)

        ax.plot(X_test, self.compute_func(model,x,y,X_test,mean_function,mean_function_test),
            marker='o', color='red', linewidth=2,label= 'Expected Improvement')
        plt.show()







    