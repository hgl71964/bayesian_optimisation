import torch as tr
import copy


class evolutionary_strategy:
    """parallelising need multiple workers"""
    def __init__(
                self,
                **kwargs,
                ):
        self.device = kwargs.get("device", "cpu")
        self.population_size = kwargs.get("population_size", 100)
        self.std = kwargs.get("std", 1e-1)
        self.lr = kwargs.get("lr", 1e-3)

    def outer_loop(
                self,
                T: int,  # iteration to run
                x0: tr.Tensor,  # initial position; shape (2,)
                r0: float,  # unormalised reward
                api: callable,  # return functional evaluation
                ):
        input_dim = x.shape[-1]
        x, y = tr.zeros((1+T, input_dim)), tr.zeros((1+T, ))
        x[0], y[0] = x0, api(x0, r0, self.device).flatten() 

        x0 = x0.view(1, -1)

        for i in range(1,T+1):

            query = x0.repeat(self.population_size, 1)  # shape:(population_size, input_dim)
            gause_noise = tr.normal(0, self.std, (self.population_size, input_dim))
            query += gause_noise

            reward = api(query, r0, self.device).flatten() # shape:(population_size, )
            avg = (reward - tr.mean(reward)) / tr.std(reward)
            x0 += x0 + self.lr /(self.population_size*self.std) * (gause_noise.T@avg)

            x[i], y[i] = x0, reward.max()  # TODO: we use max() as the reward in this round?

        return x, y




class random_opt:

    def __init__(self):
        self.device = tr.device("cpu")

    def outer_loop(
                self,
                T: int,  # iteration to run
                domain: tuple,  # domain to search 
                x0: tr.Tensor,  # initial position; shape (2,)
                r0: float,  # unormalised reward
                api: callable,  # return functional evaluation
                batch_size: int,  # random search is highly parallisable
                ):
        input_dim = x.shape[-1]
        x, y = tr.zeros((1+T * batch_size, input_dim)), tr.zeros((1+T, ))
        x[0], y[0] = x0.flatten(), api(x0, r0, self.device).flatten() 

        r1, r2 = domain  # get the domain, r1 < r2

        for i in range(T):
            query = (r2 - r1) * tr.rand((batch_size, input_dim)) + r1  # uniform in [r1, r2] 
            reward = api(query, r0, self.device).flatten()  # bottleneck for random search;
            
            x[1+i * batch_size : 1 + (i+1) * batch_size], \
                y[1+i * batch_size : 1 + (i+1) * batch_size] = query, reward
                
        return x, y