import numpy as np
import torch as tr
import copy
from time import sleep
import os
import pandas as pd

class api_utils:
    @staticmethod
    def transform(api_func: callable):
        """
        wrap the api service;
            provide small number perturbation, type conversion etc.
        """

        def wrapper(x: "query; tensor -> shape[q,d]",
                    m0: float):

            x = x.numpy()
            neg_margins = [None] * x.shape[0]

            # small number perturbation
            for i in x:
                if np.equal(i.all(), 1.):  # very extreme case; has been tested
                        i -= 1e-3     
            # generally, slightly push variables off boundary         
            x[x == 1] -= 1e-6
            x[x == 0] += 1e-6

            for _ in range(5):  # handle potential network disconnection issue
                try:
                    rewards = api_func(x)
                    for i, reward in enumerate(rewards):
                        neg_margins[i] = -(reward["margin"][0] / m0)   # record normalised negative margin
                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("reward is:", reward)
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return tr.tensor(neg_margins).view(-1, 1).float()  # assume dtype == torch.float() overall

        return wrapper
    
class env:
    """
    an simulated environment; upon receiving query, give reward
    """
    @staticmethod
    def rosenbrock(query: tr.tensor):
        """
        the rosenbrock function; f(x,y) = (a-x)^2 + b(y - x^2)^2
        Global minimum: 0; at (a, a^2)
        usually a = 1, b = 100
        """
        x, y = query.flatten()  # only take as input 2-element tensor
        return (1 - x)**2 + 100 * (y - x**2)**2


        
