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
                device: str,
                ):
            """
            Returns:
                neg_rewards: [q, 1]
            """
            x = x.cpu().numpy(); q = x.shape[0]; neg_rewards = tr.empty((q, 1), dtype=tr.float32)

            for _ in range(5):  # handle potential network disconnection issue
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=q) as executor:
                        for i, r in enumerate(executor.map(api_func, x)):  # multi-threading
                            #  r: tuple; r[0] = L2-norm, r[1] = cosine similarity
                            neg_rewards[i] = -r[1]  # TODO: 1. how to normalise the rewards; 2. which metric to use

                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return neg_rewards.to(device)  # assume dtype == torch.float() overall
        
        return wrapper
    
    @staticmethod
    def init_query(
                x0: np.ndarray,  # initial queries 
                loss_func: callable, 
                size: int,  # number of parallel queries
                ):
        y0 = tr.empty((size, 1), dtype=tr.float32)

        with concurrent.futures.ThreadPoolExecutor(max_workers=size) as executor:
            for i, r in enumerate(executor.map(loss_func, 
                                x0,                     # initial queries
                                range(size),            #  index for loss function
                                [0]*size,      #  the initial iteration (list comprehension fails)
                                )):
                #  r: tuple; r[0] = L2-norm, r[1] = cosine similarity
                y0[i] = -r[1]  # TODO: determine which metric to use 
        return y0