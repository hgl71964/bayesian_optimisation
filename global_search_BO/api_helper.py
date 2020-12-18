import numpy as np
import torch
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

            return torch.tensor(neg_margins).view(-1, 1).float()  # assume dtype == torch.float() overall

        return wrapper
    
class fake_api:
    """
    some make up function to simulate the API performance
    """

    @staticmethod
    def fake_margin(Y):
        """
        This fake margin has minimum 9

        Args:
            sequential eval -> Y: the simplified table; np.1darray; shape[4k,]
            q-concurrent eval -> Y: np.ndarray; shape[q,4k];

        Returns:
            M: total margin; list
        """
        if Y.ndim == 1:
            # some packages require input as list
            Y = Y.reshape(-1, 4)

            f_val_list = [0] * 4  # only the first 4 are useful

            # deterministic random varialbes...
            np.random.seed(42)
            A = np.random.rand(*Y.shape)
            np.random.seed(None)
            b = np.arange(1, 5)

            for i in range(4):
                temp = np.power(Y[:, i] - A[:, i], 2).sum() + b[i]
                f_val_list[i] = temp

            return [np.max([f_val_list[0], f_val_list[1]]) + f_val_list[2] + f_val_list[3]]

        else:
            recorder = [None] * Y.shape[0]
            for i, y in enumerate(Y):
                y = y.reshape(-1, 4)

                f_val_list = [None] * 4

                # some make up constants
                A = np.random.rand(*y.shape)
                b = np.arange(1, 5)

                for j in range(4):
                    temp = np.power(y[:, j] - A[:, j], 2).sum() + b[j]
                    f_val_list[j] = temp
                recorder[i] = np.max([f_val_list[0], f_val_list[1]]) + f_val_list[2] + f_val_list[3]

            return recorder

class transformer:
    """
    this class provide transformation: origin problem <=> simplified problem
    """

    @staticmethod
    def x_to_y(X):
        """
        Origin -> simplification

        Args:
            X: np.2darray; shape [n,m]

        Returns:
            Y: np.2darray; shape [n,m-1]
        """

        n = X.shape[0]
        m = X.shape[1]

        # normalisation

        N = np.sum(X, axis=1).reshape(-1, 1)
        X_prime = X / N

        # change of variable
        Y = np.zeros((n, m - 1))
        accumulator = np.ones((n,))

        for i in range(m - 1):
            Y[:, i] = X_prime[:, i] / accumulator
            accumulator -= X_prime[:, i]
        return Y

    @staticmethod
    def y_to_x(Y, N):
        """
        Origin <- simplification

        Args:
            Y: np.2darray; shape [n,m-1]
            N: the number of contract for each kind; np.1darray; shape[n,]

        Returns:
            X: np.2darray; shape [n,m]
        """

        n = Y.shape[0]
        m = Y.shape[1] + 1
        X_prime = np.zeros((n, m))

        # revert change of varialbe
        accumulator = np.ones((n,))
        for i in range(m - 1):
            X_prime[:, i] = Y[:, i] * accumulator
            accumulator -= X_prime[:, i]

        X_prime[:, -1] = accumulator

        # revert normalisation
        X = X_prime * N.reshape(-1, 1)

        # round it to nearest integers???
        return np.round(X).astype(int)
