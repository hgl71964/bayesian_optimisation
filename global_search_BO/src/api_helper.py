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

        def wrapper(x: tr.tensor,  #  shape[q,d]; q query, d-dimensional
                    r0: float,  #  unormalised reward
                    ):
            """
            Returns:
                neg_margins: [q, 1]
            """
            q = x.shape[0]
            neg_margins = tr.zeros(q, )

            # we may want to push query off the boundary
            # for i in x:
                # if np.equal(i.all(), 1.):  # very extreme case; has been tested
                        # i -= 1e-3     
            ## generally, slightly push variables off boundary         
            # x[x == 1] -= 1e-6
            # x[x == 0] += 1e-6

            for _ in range(5):  # handle potential network disconnection issue
                try:
                    for i in range(q):  # TODO: apply multi-threading, rather than sequential
                        r = api_func(x)  # float
                        neg_margins[i] = -(r / r0)   # record normalised negative margin

                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return neg_margins.view(-1, 1).float()  # assume dtype == torch.float() overall

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
        return tr.tensor([(1 - x)**2 + 100 * (y - x**2)**2])


        