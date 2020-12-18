"""
2020 Summer internship

This script implement kownledge gradient as acquisition function

This implementation is based on 'A Tutorial on Bayesian Optimization' by Peter I. Frazier

We refer as 'the paper', and we use his notations

"""

import matplotlib.pyplot as plt
import numpy as np
import copy

class _knowledge_gradient:
    """
    knowledge gradient used by bayes opter

    have two callable functions:

    1. compute_func: used to compute the values of function

    2. propose_position: returns the maximum of acquisition function and its x

    """

    def __init__(self,**kwargs):
        self.T = kwargs.get("T")
        self.J = kwargs.get("J")
        self.R = kwargs.get("R")
        self.a = kwargs.get("a")

    def compute_func(self):
        pass

    def propose_position(self):
        pass

    def _simulation(self,model,x,y,X_test,mu_star,mean_function,mean_function_test):
        """
        Algorithm 2 in the paper

        Notations:
            mu_star = mu^*_n
            mu_plus = mu^*_n+1
            y_plus = y_{n+1}

        Goal:
            evaluate KG(x) at a given x via Monte Carlo Simulation

        Args:
            x: existing samples; shape[m,n]
            y: existing sample values; shape[m,]

            mu_star: float, previously evaluated largest value
            X_test: a given location for evaluation; np -> shape [1,n]

        Returns:

        """
        
        #pst_μ, pst_σ are scalars, since we have one test point
        pst_μ, pst_Σ=model.gp_posterior(x,y,X_test,mean_function,mean_function_test)
        pst_σ = np.sqrt(np.diag(pst_Σ)).flatten()

        for j in range(self.J):

            Z = np.random.rand(0,1)
            y_plus = Z * pst_σ + pst_μ # scalar

            """
            x_sample: shape [m+1,n] (with the new observation)
            y_sample: shape [m+1,] (with the new observation)
            """
            x_new_sample = np.concatenate((x,X_test),axis=0)
            y_new_sample = np.append(y,y_plus)

            """
            Now we need to optimise with this new observation
            """

        return 

    def _stochastic_gradient(self):
        pass

