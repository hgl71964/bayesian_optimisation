import numpy as np
import torch as tr
import copy
from time import sleep
import os
import concurrent.futures
import multiprocessing
import asyncio

class api_utils:

    @staticmethod
    def wrapper(api_func: callable):  """IO bound"""
        def wrapper(x: tr.tensor,  #  shape[q,d]; q query, d-dimensional
                    r0: float,  #  unormalised reward
                    device: str,
                    ):
            """
            Returns:
                neg_rewards: [q, 1]
            """
            x = x.cpu(); q = x.shape[0]; neg_rewards = tr.zeros((q, ))

            for _ in range(5):  # handle potential network disconnection issue
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        for i, r in enumerate(executor.map(api_func, x)):  # multi-threading
                            neg_rewards[i] = -(r/r0)   

                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return neg_rewards.view(-1, 1).to(device)  # assume dtype == torch.float() overall

        return wrapper