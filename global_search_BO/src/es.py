import torch as tr
import copy


class evolutional_strategy:

    def __init__(self):
        pass


    def outer_loop(
                self,
                ):

        
        return 


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