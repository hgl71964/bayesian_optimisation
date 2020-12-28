import numpy as np
import torch as tr
import copy
import os
import concurrent.futures

class api_utils:

    @staticmethod
    def wrapper(api_func: callable):  #  IO bound
        def wrapper(x: tr.tensor,  #  shape[q,d]; q query, d-dimensional
                r0: float,  #  unormalised reward
                iteration: int,  
                device: str,
                ):
            """
            Returns:
                neg_rewards: [q, 1]
            """
            x = x.cpu().numpy(); q = x.shape[0]; neg_rewards = tr.empty((q, 1), dtype=tr.float32); 

            for _ in range(5):  # handle potential network disconnection issue
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=q) as executor:
                        for i, r in enumerate(executor.map(api_func, 
                                                        x,
                                                        range(q),
                                                        [iteration] * q,  
                                                        )): 

                            #  r: tuple; r[0] = L2-norm, r[1] = cosine similarity
                            neg_rewards[i] = -r[1]  # TODO: 1. how to normalise the rewards; 2. which metric to use

                except TypeError as ter:
                    print(f"api has error {ter}"); print("query is:", repr(x)); sleep(10)
                else:
                    break

            return neg_rewards.to(device)  # assume dtype == torch.float() overall
        
        return wrapper
    
    @staticmethod
    def init_reward(x0: np.ndarray, loss_func: callable):
        q = x0.shape[0]; y0 = tr.empty((q, 1), dtype=tr.float32); 

        with concurrent.futures.ThreadPoolExecutor(max_workers=q) as executor:
            for i, r in enumerate(executor.map(loss_func, 
                                                x0,             
                                                range(q),       #  index 
                                                [0]*q,          #  0-th iteration (list comprehension fails)
                                                )):
                #  r: tuple; r[0] = L2-norm, r[1] = cosine similarity
                y0[i] = -r[1]  # TODO: determine which metric to use 
        return y0

    @staticmethod
    def init_query(size, search_bounds):
        """make initial query"""
        x0 = np.empty((size, len(search_bounds)))
        for i in range(size):
            for j, r in enumerate(search_bounds):
                x0[i][j] = np.random.uniform(search_bounds[j][0], search_bounds[j][1])
        return x0