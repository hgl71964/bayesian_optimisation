"""
2020 Summer internship

a api_helper allows:
            origin problem <=> simplified problem
            assuming m accounts & n kinds of contracts
            provide wrapper for the actual api function

Details see my thesis
"""
import numpy as np
import torch
import copy
from time import sleep
import os
import pandas as pd


class portfolio_constructor:
    
    @staticmethod
    def construct_portfolio(tables: list, folder_name, file_name= "") -> pd.DataFrame:
        """
        Args:
            tables: a list of table; each table is a hand-crafted portfolio
        """
        dfs = [None] * len(tables)

        #  assume the tables has the same order
        contracts = ["BTSU0", "DUU0", "IKU0", "OATU0", "OEU0", "RXU0", "UBU0"]
        accounts = ["Barclays - F&O Clearing", 
                    "Barclays - 3952C F&O", 
                    "BAML - F&O Clearing"]
        for i,table in enumerate(tables):

            assert (len(table) == len(contracts) and len(table[0]) == len(accounts))
            dfs[i] = pd.DataFrame(table, index = contracts, columns = accounts)

        if not file_name:
            print("not saving unless provide a file_name")

        else:
            full_path = os.path.join(folder_name, file_name)
            with open(full_path + ".pt", "wb") as f:
                torch.save((dfs), f)
                f.close()

        return dfs


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
    
    @staticmethod
    def create_start_position(wrap_class: callable, 
                            cobdate: "datetime",
                            portfolio: pd.DataFrame,
                            margins_param: dict = {},
                            fixed_start = False,
                            ) -> tuple:

        #  instantiate api from a given portfolio
        wrap = wrap_class(cobdate, portfolio, randomize = False, margins = margins_param) 
        ndim = wrap.x0.shape[0]

        if fixed_start:
            x0 = np.ones(ndim,) * 0.5
            margins_init = wrap.f(x0)
        else:
            x0 = copy.deepcopy(wrap.x0)
            margins_init = wrap.f(wrap.x0)
            
        m0 = margins_init['margin'][0] # float
        b0 = margins_init['buffer'][0]
        print(f"${m0:,.0f}, buffer is: ${b0:,.0f}")

        x0 = torch.from_numpy(x0).float().view(-1, ndim)  # shape [n,d]
        y0 = torch.tensor([-1], dtype=torch.float).view(-1, 1)  # shape [n,1]; min Margin == max (-Margin)
        return (wrap, x0, y0, m0)

    @staticmethod
    def create_random_start(wrap_class: "opt_class", 
                            cobdate: "date",
                            positions: "dataframe", 
                            folder_name: str,
                            n: int = 3, 
                            ):
        """
        Args:
            n: numbers of starting points; = runs of the experiment
            random: if True -> randomly initialise positions
        Returns:
            data: list[tuple]
        """
        data = [None] * n
        for i in range(n):   # if randomize, function structure change even within the same day
            if i == 0:
                wrap = wrap_class(cobdate, positions, randomize = False) 
                margins_init = wrap.f(wrap.x0)
                x0 = copy.deepcopy(wrap.x0)
                ndim = wrap.x0.shape[0]  # dimensions of inputx
            else:
                x0 = np.random.rand(ndim)
                margins_init = wrap.f(x0)

            m0 = margins_init['margin'][0] # float
            b0 = margins_init['buffer'][0]
            print(f"${m0:,.0f}, buffer is: ${b0:,.0f}")

            # initial samples; 1 data point; ndim-dimensional input; assume data type = torch.float
            x0 = torch.from_numpy(x0).float().view(-1, ndim)  # shape [n,d]
            y0 = torch.tensor([-1], dtype=torch.float).view(-1, 1)  # shape [n,1]; min Margin == max (-Margin)
            data[i] = (x0, y0, m0)
        
        if folder_name:
            full_path = os.path.join(folder_name, "random_start_data.pt")
            with open(full_path, "wb") as f:
                torch.save(data, f)
                f.close()
        return data

    @staticmethod
    def load_random_start(folder_name: str):
        full_path = os.path.join(folder_name, "random_start_data.pt")
        with open(full_path, "rb") as f:
            objs = torch.load(f)
            f.close()
        return objs

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
