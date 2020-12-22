import numpy as np
import torch as tr
import copy
from time import sleep


class random_opt:

    def __init__(self):
        self.device = tr.device("cpu")

    def outer_loop(
                self,
                T: int,  # iteration to run
                x0: tr.Tensor,  # initial position; shape (2,)
                r0: float,  # unormalised reward,
                api: callable,  # return functional evaluation
                batch_size: int,  # random search is highly parallisable
                ):
        input_dim = x.shape[-1]
        x, y = tr.zeros((T * batch_size, input_dim)), tr.zeros((T, ))
        x0 = x0.flatten() 

        for i in range(T):
            x[i], y[i] = x0, api(x0, r0, self.device)

            query = tr.rand(batch_size, input_dim)
            reward = api(query, r0, self.device)  # bottleneck for random search
            
            x[i:i+batch_size], y[] = 

        return x, y

class ADAM_opt:

    def __init__(self, **kwargs):
        self.beta1 = kwargs.get(["beta1"], .9)
        self.beta2 = kwargs.get(["beta2"], .999)
        self.alpha = kwargs.get(["alpha"], 1e-3) 
        self.eta = kwargs.get(["eta"], 1e-8)
        self.mu = tr.zeros((2,))
        self.v = tr.zeros((2,))
        self.counter = 0
        self.device = tr.device("cpu")

    def outer_loop(
                self,
                T: int,  # iteration to run
                x0: tr.Tensor,  # initial position; shape (2,)
                r0: float,  # unormalised reward,
                api: callable,  # return functional evaluation
                api_grad: callable,  # return gradient 
                ):
        input_dim = x.shape[-1]
        x, y = tr.zeros((T, input_dim)), tr.zeros((T, ))
        x0 = x0.flatten()

        for i in range(T):

            # collect stats
            self.counter += 1
            x[i], y[i] = x0, api(x0, r0, self.device).flatten()

            # query for gradient 
            grad = self.api_grad(x0)  # grad; shape(2, )

            # momentum 
            self.mu = self._moment_update(1, self.beta1, grad, self.mu)
            self.v = self._moment_update(2, self.beta2, grad, self.v)

            # descent
            up = self.alpha * (self.mu / (1- (self.beta1 ** self.counter)) )
            down = self.eta + tr.sqrt(self.v / (1- (self.beta2**self.counter)) )
            x0 -= up/down

        return x, y
    
    def _moment_update(self, order, decay, grad, moment):
        return (1-decay) * (grad ** order) + decay * moment





