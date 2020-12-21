import numpy as np
import torch as tr
import copy
from time import sleep


class ADAM_opt:

    def __init__(self, **kwargs):
        self.beta1 = kwargs.get(["beta1"], .9)
        self.beta2 = kwargs.get(["beta2"], .999)
        self.alpha = kwargs.get(["alpha"], 1e-3) 
        self.eta = kwargs.get(["eta"], 1e-8)
        self.mu = np.zeros((2,))
        self.v = np.zeros((2,))
        self.counter = 0


    def outer_loop(
                self,
                T: int,  # iteration to run
                x: np.ndarray,  # initial position 
                r0: float,  # unormalised reward,
                api: callable,  # return functional evaluation
                api_grad: callable,  # return gradient 

                ):
        

        for i in range(T):

            # collect stats
            self.counter += 1


            grad = self.api_grad()

        return x, y
    
    def _moment_update(self):





