"""
2020 Summer internship

slice a function; show pair-wise structure; holding the other fixed
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time

class slicer:

    def __init__(self, total_dim):
        self.total_dim = total_dim

    @staticmethod
    def static_random_query(total_dim: int,
                            api: callable,
                            m0: float,
                            N: int = 400,
                            q: int = 10,
                            hard_termination=False,
                            ):
        #  static method of random_query
        if not hard_termination:
            for i in range(0, N, q):
                random_query = torch.rand(q, total_dim)
                rs = api(random_query, m0)

                if i % 50 == 0:
                    print(f"{i} queries are made")
                
                if i == 0:
                    X = random_query
                    y = rs
                else:
                    X = torch.cat([X, random_query], dim=0)
                    y = torch.cat([y, rs], dim=0)
        else:
            i=0
            start_time = time.time()  # ignore N but set 20 mins hard termination
            while True:
                random_query = torch.rand(q, total_dim)
                rs = api(random_query, m0)

                if (i*q) % 1000 == 0:
                    print(f"{i*q} queries are made")
                
                if i == 0:
                    X = random_query
                    y = rs
                else:
                    X = torch.cat([X, random_query], dim=0)
                    y = torch.cat([y, rs], dim=0)
                i+=1

                if (time.time() - start_time)/60 > 20:
                    break
        return X, y


    def random_query(self,
                     api: callable,
                     m0: float,
                     N: int,
                     q: int = 10):
        """
        generate random queries

        Args:
            N: number of random query
            q: parallel proposal

        Returns:
            X, y: query, reward
        """

        for i in range(0, N, q):
            print(f"query {i}")
            random_query = torch.rand(q, self.total_dim)

            time1 = time.time()
            rs = api(random_query, m0)
            time2 = time.time()
            print(f"api took {(time2-time1):.1f}s")

            if i == 0:
                X = random_query
                y = rs
            else:
                X = torch.cat([X, random_query], dim=0)
                y = torch.cat([y, rs], dim=0)
        return X, y

    def slice_query(self,
                    api: callable,
                    m0: float,
                    interest_dim: list,
                    num_main_dim: int = 12,
                    num_other_dim: int = 3):
        """
        Args:
            api: reward = api(query, m0); reward shape [len(query),1]
            m0: initial margin
            interest_dim: list[int]; specify two dimension that is interesting
            num_main_dim: number of samples along interesting dimensions
            num_other_dim: number of samples size for other dimensions; (between 0 and 1)

        Returns:
            slices: list[tuple]; tuple -> (query, reward); number of slices == np.linspace(0, 1, num_other_dim)
        """

        fix_dims = torch.linspace(0., 1., num_other_dim, dtype = torch.float)
        slices = [None] * len(fix_dims)

        for i, dim_vals in enumerate (fix_dims):

            print(f"runs {i+1}")

            x = torch.linspace(0., 1., num_main_dim, dtype = torch.float)
            y = torch.linspace(0., 1., num_main_dim, dtype = torch.float)
            X, Y = torch.meshgrid(x, y)
            rewards = torch.zeros_like(X, dtype = torch.float)

            for j in range(X.size(0)):

                query = torch.full((X.size(-1), self.total_dim), dim_vals, dtype = torch.float)
                query[:, interest_dim] = torch.cat([X[j,:].view(-1, 1), Y[j,:].view(-1, 1)], dim=1)
                rs = api(query, m0)
                rewards[j,:] = -1 *  rs.flatten()

            slices[i] = (X, Y, rewards, dim_vals, m0, interest_dim)

        return slices

    def plot(self,
            slices,
            index: int,
            color_map_range: np.ndarray = np.arange(.5, 1.1, 0.05)):
        """
        visualise pair-wise function structure;
        Note: the shown margins are normalised by initial margin,
                                so the lower it shows, the less margin we need to pay

        Args:
            slices: returned by self.slice_query
            index: plot the index-th data in slices
        """
        X, Y, M, dim_vals, m0, interest_dim = slices[index]

        fig, ax = plt.subplots(figsize=(12,8))
        cf = ax.contourf(X, Y, M, color_map_range, extend = 'both')

        ax.set_title(f"Pairwise function structure", fontsize = 16)
        ax.set_ylabel(f"dim {interest_dim[1]}", fontsize = 14)
        ax.set_xlabel(f""" dim {interest_dim[0]}
        The initial margin is ${m0:,.0f}, and other dimensions have the same values {dim_vals} """, fontsize = 14)

        ax.set_xlim(left = 0, right = 1)
        ax.set_ylim(bottom = 0, top = 1)
        cbar = fig.colorbar(cf, ax = ax)
        cbar.ax.tick_params(labelsize = 12)

        plt.show()
        return None

    def plot_4_figs(self,
            slices,
            color_map_range: np.ndarray = np.arange(.5, 1.1, 0.05)):
        """
        visualise pair-wise function structure;
        Note: the shown margins are normalised by initial margin,
                                so the lower it shows, the less margin we need to pay

        Args:
            slices: returned by self.slice_query
            index: plot the index-th data in slices
        """
        fig, ax = plt.subplots(nrows =2 , ncols = 2, figsize=(14,12))
        ax = ax.flatten()

        for index in range(4):
            X, Y, M, dim_vals, _, interest_dim = slices[index]
            subax = ax[index]

            cf = subax.contourf(X, Y, M, color_map_range, extend = 'both')
            subax.set_ylabel(f"dim {interest_dim[1]}", fontsize = 14)
            subax.set_xlabel(f"dim {interest_dim[0]}, other dim: {dim_vals:.2f}", fontsize = 14)
            subax.set_xlim(left = 0, right = 1)
            subax.set_ylim(bottom = 0, top = 1)
            cbar = fig.colorbar(cf, ax = subax)
            cbar.ax.tick_params(labelsize = 12)

        plt.show()
        return None

    def save_data(self, slices, folder_name, save_name = ""):
        if save_name:
            full_path = os.path.join(folder_name, save_name)
            with open(full_path + ".pt", "wb") as f:
                torch.save(slices, f)
                f.close()
                print("save successful")
        else:
            print("give a file name!")
            raise Exception("The file name is empty")

    def load_data(self, folder_name, save_name = ""):
        if save_name:
            full_path = os.path.join(folder_name, save_name)
            with open(full_path + ".pt", "rb") as f:
                objs = torch.load(f)
                f.close()
                return objs
        else:
            raise Exception("The file name is empty")
