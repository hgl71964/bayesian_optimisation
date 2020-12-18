"""
2020 Summer internship

help to visualise experiment results && other utils
"""

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from api_helper import portfolio_constructor
import pandas as pd

class exp_plotter:

    @staticmethod
    def display_portfolio(folder_name: str,
                            file_name: str,
                            number: int, #  portfolio number
                            ):
        
        full_path = os.path.join(folder_name, file_name)

        with open(full_path + ".pt", "rb") as f:
            all_portfolios = torch.load(f)
            f.close()
        portfolio, margin_params = all_portfolios[number]

        dfs = portfolio_constructor.construct_portfolio([portfolio], "", "")

        return dfs[0], margin_params
    
    @staticmethod
    def gp_evaluation(folder_name: str,
                        file_name: str,
                        ):
        full_path = os.path.join(folder_name, file_name)

        with open(full_path + ".pt", "rb") as f:
            all_gp_res = torch.load(f)
            f.close()

        print("according to log marginal likelihood, the best gp is:")
        print("")
        temp = sorted(all_gp_res, key=lambda  x: x[-1])
        print(temp[-1])
        return all_gp_res
    
    @staticmethod
    def acq_func_portfolio(folder_name: str,
                            file_names: list,  #  provide a list of results files
                            ):
        res = {}
        for file_name in file_names:
            full_path = os.path.join(folder_name, file_name)

            with open(full_path + ".pt", "rb") as f:
                temp = torch.load(f)
                f.close()
            for i in temp:  # temp -> dict
                res[i] = temp[i]
 
        temp = -float("inf")
        for i in res:
            for y in res[i][1]:
                if y.max() > temp:
                    temp = y.max()
                    temp_acq = i
                    temp_margin = res[i][2]
        print(f"best acquisition function: {temp_acq}")
        print(f"best drop %: {(1+temp.item())*100:.3f}%")
        print(f"starting margin ${(temp_margin):,.0f}")
        print(f"lowest margin ${(-temp.item() * temp_margin):,.0f}")
        print(f"saving margin ${(temp_margin*(1+temp.item())):,.0f}")
        return res
    
    @staticmethod
    def acq_func(res: dict,
                acq_names: list,  
                ):
        performance = []
        for acq_name in acq_names:
            x = res[acq_name][0][0]
            y = res[acq_name][1][0]
            drop = (1+y.max().item())*100
            performance.append((acq_name, drop))
        return performance

    @staticmethod
    def convergent_plots(res, #  res returned by acq_func_portfolio
                acq_func_names: list, #  list of name of an acquisition 
                turning_point: int, #  the turning point for warmup
                y_axis_lim :list, #  to set y_axis_lim to the normalised plot
                ):
        
        parallel = ["qUCB", "qEI"]
        warmup = ["warmup1", "warmup2"]
        _, ax = plt.subplots(nrows =1 , ncols = 2, figsize=(14,6))
        ax1, ax2 = ax
        
        for acq_func_name in acq_func_names:
            plot_warmup = False
            ans = res[acq_func_name]
            starting_margin = ans[2]

            y = []

            if acq_func_name[:4] in parallel or acq_func_name[:3] in parallel:
                for i in range(0, ans[1][0].size(0), 2):
                    temp = ans[1][0][i:i+2].max().item()
                    y.append(temp)

            elif acq_func_name in warmup:
                iteration = min(turning_point, ans[1][0].size(0))
                if iteration == ans[1][0].size(0):
                    print("iteration hasn't reached turning point...")
                else:
                    plot_warmup = True

                
                for i in range(0, ans[1][0].size(0)):
                    y.append(ans[1][0][i])
            
            else:
                y = [ans[1][0][i].item() for i in range(0, ans[1][0].size(0))]
            
            margins = [-i*starting_margin for i in y]

            ax1.set_title("convergent plot")
            ax1.plot([i+1 for i in range(len(y))], [1+i for i in y], label = f"{acq_func_name} path")

            ax2.set_title("actual margin")
            ax2.plot([i+1 for i in range(len(y))], margins, label = f"{acq_func_name} path")

            ax1.set_xlabel("iteration")
            ax1.set_ylabel("reward")
            ax2.set_xlabel("iteration")
            ax2.set_ylabel("margins")

        if y_axis_lim:
            ax1.set_ylim(bottom = y_axis_lim[0], top=y_axis_lim[1])
            b, u = y_axis_lim[0] * starting_margin, y_axis_lim[1] * starting_margin
            ax2.set_ylim(bottom = b, top=u)
        if plot_warmup:
            ax1.axvline(turning_point,linewidth = 1,alpha=0.6,label="turning_point")
            ax2.axvline(turning_point,linewidth = 1,alpha=0.6,label="turning_point")
        ax1.legend()
        ax2.legend()
        plt.show()
        return None
    

    @staticmethod
    def convergent_plot(res, #  res returned by acq_func_portfolio
                acq_func_name: str, #  name of an acquisition 
                turning_point: int, #  the turning point for warmup
                y_axis_lim :list, #  to set y_axis_lim to the normalised plot
                ):
        
        parallel = ["qUCB", "qEI"]
        warmup = ["warmup1", "warmup2"]
        plot_warmup = False
        ans = res[acq_func_name]
        starting_margin = ans[2]

        y = []

        if acq_func_name[:4] in parallel or acq_func_name[:3] in parallel:
            for i in range(0, ans[1][0].size(0), 2):
                temp = ans[1][0][i:i+2].max().item()
                y.append(temp)

        elif acq_func_name in warmup:
            iteration = min(turning_point, ans[1][0].size(0))
            if iteration == ans[1][0].size(0):
                print("iteration hasn't reached turning point...")
            else:
                plot_warmup = True

            
            for i in range(0, ans[1][0].size(0)):
                y.append(ans[1][0][i])
        
        else:
            y = [ans[1][0][i].item() for i in range(0, ans[1][0].size(0))]
        
        margins = [-i*starting_margin for i in y]
        
        _, ax = plt.subplots(nrows =1 , ncols = 2, figsize=(14,6))
        ax1, ax2 = ax

        ax1.set_title("normalised margin")
        ax1.plot([i+1 for i in range(len(y))], [-i for i in y], color = "olive", label = "normalised margin path")

        ax2.set_title("actual margin")
        ax2.plot([i+1 for i in range(len(y))], margins, color = "red", label = "saving margin path")

        ax1.set_xlabel("iteration")
        ax1.set_ylabel("margins %")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("margins")

        if plot_warmup:
            ax1.axvline(turning_point,linewidth = 1,alpha=0.6,label="turning_point")
            ax2.axvline(turning_point,linewidth = 1,alpha=0.6,label="turning_point")

        
        if y_axis_lim:
            ax1.set_ylim(bottom = y_axis_lim[0], top=y_axis_lim[1])
            b, u = y_axis_lim[0] * starting_margin, y_axis_lim[1] * starting_margin
            ax2.set_ylim(bottom = b, top=u)

        plt.show()
        return None
    
    @staticmethod
    def optimal_allocation(res,#  res returned by acq_func_portfolio
                            starting_portfolio, #  DataFrame of starting portfolio
                            ):
        temp = -float("inf")
        for i in res:
            for y in res[i][1]:
                y_flatten = y.flatten()
                if y.max() > temp:

                    temp = y.max()
                    _ = y.max().item()
                    _ = i
                    _ = res[i][2]
                    index = y_flatten.argmax()
                    best_x = res[i][0][0][index]
        
        #  transform best_x back to portfolio table
        (n, m) = starting_portfolio.shape
        best_x = best_x.numpy().reshape(n, m-1)

        N = starting_portfolio.to_numpy().sum(axis=1)
        X_prime = np.zeros((n,m))

        #  revert change of varialbe
        accumulator = np.ones((n,))
        for i in range(m - 1):
            X_prime[:, i] = best_x[:, i] * accumulator
            accumulator -= X_prime[:, i]
        X_prime[:, -1] = accumulator
        X = X_prime * N.reshape(-1, 1)

        return pd.DataFrame(data=np.round(X).astype(int), index=starting_portfolio.index, columns=starting_portfolio.columns)
    
    @staticmethod
    def query2portfolio(res,#  res returned by acq_func_portfolio
                        starting_portfolio, #  DataFrame of starting portfolio
                        acq_func_name: str, 
                        ):
        ans = res[acq_func_name]
        temp = -float("inf")
        for y in ans[1]:
            y_flatten = y.flatten()
            if y.max() > temp:
                temp = y.max()
                index = y_flatten.argmax()
                best_x = ans[0][0][index]

        #  transform best_x back to portfolio table
        (n, m) = starting_portfolio.shape
        best_x = best_x.numpy().reshape(n, m-1)

        N = starting_portfolio.to_numpy().sum(axis=1)
        X_prime = np.zeros((n,m))

        #  revert change of varialbe
        accumulator = np.ones((n,))
        for i in range(m - 1):
            X_prime[:, i] = best_x[:, i] * accumulator
            accumulator -= X_prime[:, i]
        X_prime[:, -1] = accumulator
        X = X_prime * N.reshape(-1, 1)

        return pd.DataFrame(data=np.round(X).astype(int), index=starting_portfolio.index, columns=starting_portfolio.columns)

    @staticmethod
    def compare_optimal_query(res):
        ans = []

        for i in res:
            temp = -float("inf")
            for y in res[i][1]:
                y_flatten = y.flatten()
                if y.max() > temp:

                    temp = y.max()
                    index = y_flatten.argmax()
                    best_x = res[i][0][0][index]

            ans.append((i, best_x, -temp.item()))
        return ans
